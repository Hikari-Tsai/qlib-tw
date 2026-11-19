"""把 data/raw/ 的證交所原始 JSON 轉成 Qlib 資料集。

步驟：
1. 解析每日 MI_INDEX → 普通股日資料面板（data/processed/panel.parquet）
2. 除權息／減資／面額變更事件 → 還原比率；2011 年前減資等表上沒有的事件，以「漲跌價差」反推的參考價補上
3. 每檔計算 factor（還原價＝原始價×factor，上市首日 factor=1），輸出每檔 CSV 到 data/csv/
4. 加權指數 TWII（來自 ../data/taiex_daily.csv，僅讀取）
5. 呼叫 scripts/dump_bin.py 產生 qlib_data/，再重寫 instruments（all.txt 不含 TWII、tw100.txt 每月前 100 名）
6. 資料品質報告 data/quality_report.json

用法：uv run python build_dataset.py [--skip-parse]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# 路徑可用環境變數覆寫（GitHub Actions 每日更新時用近期資料的工作目錄）
RAW = Path(os.environ.get("QTW_RAW") or ROOT / "data" / "raw")
PROCESSED = Path(os.environ.get("QTW_PROCESSED") or ROOT / "data" / "processed")
CSV_DIR = Path(os.environ.get("QTW_CSV") or ROOT / "data" / "csv")
QLIB_DIR = Path(os.environ.get("QTW_QLIB") or ROOT / "qlib_data")
TAIEX_CSV = Path(os.environ.get("QTW_TAIEX_CSV") or ROOT.parent / "data" / "taiex_daily.csv")
REPORT_PATH = Path(os.environ.get("QTW_REPORT") or ROOT / "data" / "quality_report.json")
BENCHMARK = "TWII"

# MI_INDEX 欄位名稱 → 內部名稱（依名稱對應，不依位置）
DAILY_FIELDS = {
    "證券代號": "code", "證券名稱": "name", "成交股數": "volume", "成交筆數": "trades",
    "成交金額": "amount", "開盤價": "open", "最高價": "high", "最低價": "low",
    "收盤價": "close", "漲跌(+/-)": "sign", "漲跌價差": "change_pts",
}
COMMON_STOCK = re.compile(r"^[1-9]\d{3}$")  # 4 碼且非 0 開頭（排除 00xx ETF／受益憑證）
LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")  # 漲跌幅由 7% 放寬為 10%
POOLS = {"tw100": 100, "tw300": 300}  # 股票池名稱 → 每月取成交金額前 N 名
POOL_LOOKBACK = 60  # 股票池：月底往回 60 個交易日（約一季）的平均成交金額，降低雜訊換手


def to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s.where(~s.isin(["--", "", "---", "----", "X0.00"])), errors="coerce")


def roc_to_date(s: str) -> pd.Timestamp:
    nums = re.findall(r"\d+", s)
    return pd.Timestamp(int(nums[0]) + 1911, int(nums[1]), int(nums[2]))


def load_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 1. 日資料
def parse_daily_file(path: Path) -> pd.DataFrame | None:
    j = load_gz(path)
    tables = j.get("tables") or []
    # 舊版回傳格式可能是 fieldsN/dataN，一併相容
    for k in j:
        if k.startswith("fields") and isinstance(j[k], list):
            tables.append({"fields": j[k], "data": j.get("data" + k[6:], [])})
    for t in tables:
        fields = t.get("fields") or []
        if "證券代號" in fields and "收盤價" in fields:
            missing = set(DAILY_FIELDS) - set(fields)
            if missing:
                raise ValueError(f"{path.name} 缺少欄位 {missing}")
            df = pd.DataFrame(t["data"], columns=fields)[list(DAILY_FIELDS)].rename(columns=DAILY_FIELDS)
            df.insert(0, "date", pd.Timestamp(path.name[:8]))
            return df
    return None


def build_panel() -> pd.DataFrame:
    files = sorted((RAW / "daily").glob("*/*.json.gz"))
    frames, n_trade_days = [], 0
    for i, p in enumerate(files):
        df = parse_daily_file(p)
        if df is None:
            continue
        n_trade_days += 1
        frames.append(df[df["code"].str.strip().str.match(COMMON_STOCK)])
        if i % 500 == 0:
            print(f"  解析 {p.name} ({i}/{len(files)})", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    raw["code"] = raw["code"].str.strip()
    raw["name"] = raw["name"].str.strip()
    raw = raw[~raw["name"].str.contains("DR", regex=False)]  # 排除 4 碼的存託憑證（如 9105 泰金寶-DR）
    sign_txt = raw["sign"].astype(str).str.replace(r"<[^>]*>", "", regex=True).str.strip()
    raw["sign"] = np.select([sign_txt == "+", sign_txt == "-", sign_txt.str.upper() == "X"], [1.0, -1.0, np.nan], 0.0)
    raw["sign_x"] = sign_txt.str.upper() == "X"  # 不比價
    for c in ["volume", "trades", "amount", "open", "high", "low", "close", "change_pts"]:
        raw[c] = to_num(raw[c])
    raw = raw.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  交易日 {n_trade_days}，普通股列數 {len(raw):,}，股票數 {raw['code'].nunique()}")
    return raw


# ---------------------------------------------------------------- 2. 事件
def load_events() -> pd.DataFrame:
    specs = {
        "TWT49U": ("資料日期", "股票代號", "除權息前收盤價", "除權息參考價"),
        "TWTAUU": ("恢復買賣日期", "股票代號", "停止買賣前收盤價格", "恢復買賣參考價"),
        "TWTB8U": ("恢復買賣日期", "股票代號", "停止買賣前收盤價格", "恢復買賣參考價"),
    }
    rows = []
    for name, (fd, fc, fp, fr) in specs.items():
        for p in sorted((RAW / "events").glob(f"{name}_*.json.gz")):
            j = load_gz(p)
            fields = j.get("fields") or []
            if not j.get("data"):
                continue
            idx = {f: fields.index(f) for f in (fd, fc, fp, fr)}
            for r in j["data"]:
                rows.append((name, roc_to_date(r[idx[fd]]), str(r[idx[fc]]).strip(), r[idx[fp]], r[idx[fr]]))
    ev = pd.DataFrame(rows, columns=["source", "date", "code", "prev_close", "ref"])
    ev["prev_close"] = to_num(ev["prev_close"])
    ev["ref"] = to_num(ev["ref"])
    ev = ev[ev["code"].str.match(COMMON_STOCK)].dropna(subset=["prev_close", "ref"])
    ev = ev[(ev["prev_close"] > 0) & (ev["ref"] > 0)].drop_duplicates(["source", "date", "code"])
    ev["ratio"] = ev["ref"] / ev["prev_close"]
    return ev.reset_index(drop=True)


# ---------------------------------------------------------------- 3. factor 與 CSV
def tick_size(p: np.ndarray) -> np.ndarray:
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def compute_stock(df: pd.DataFrame, ev: pd.DataFrame, implied_tol: float) -> tuple[pd.DataFrame, list[dict]]:
    """df：單一股票有成交的日資料（依日期排序）。回傳加上 factor 等欄位的資料與採用的事件清單。"""
    df = df.copy()
    prev_close = df["close"].shift(1)
    ref_implied = df["close"] - df["sign"] * df["change_pts"]  # 不比價(X) 時 sign 為 NaN → 參考價未知
    df["ref"] = ref_implied
    implied_ratio = ref_implied / prev_close

    ratio = pd.Series(1.0, index=df.index)
    used = []
    dates = df["date"].to_numpy()
    # 表列事件：套用在事件日當天或之後第一個有成交的交易日
    for (d, sub) in ev.groupby("date"):
        pos = np.searchsorted(dates, np.datetime64(d))
        if pos >= len(df) or pos == 0:
            continue
        r = float(np.prod(sub["ratio"]))
        i = df.index[pos]
        ratio[i] *= r
        used.append({"date": d, "ratio": r, "source": "+".join(sub["source"]),
                     "implied_ratio": float(implied_ratio[i]) if pd.notna(implied_ratio[i]) else None,
                     "prev_close_table": float(sub["prev_close"].iloc[0]),
                     "prev_close_panel": float(prev_close[i])})
    # 表上沒有、但漲跌價差反推的參考價與前一日收盤差距過大 → 視為未列表事件（多為 2011 年前減資）
    table_days = {u["date"] for u in used}
    cand = implied_ratio[(implied_ratio - 1).abs() > implied_tol]
    for i, r in cand.items():
        d = df.at[i, "date"]
        if any(abs((d - t).days) <= 7 for t in table_days):
            continue
        ratio[i] *= r
        used.append({"date": d, "ratio": float(r), "source": "implied", "implied_ratio": float(r),
                     "prev_close_table": None, "prev_close_panel": float(prev_close[i])})
    # 不比價(X) 日沒有參考價可反推；若相對前一收盤的漲跌超過當時漲跌幅上限，單日交易不可能做到，
    # 視為表上查不到的公司行為（多為 2011 年前減資、分割），以當日收盤為參考價（當日還原報酬記為 0）
    lim_day = np.where(df["date"] >= LIMIT_CHANGE_DATE, 0.10, 0.07) + 0.005
    raw_ret = df["close"] / prev_close - 1
    x_jump = df["sign_x"] & (raw_ret.abs() > lim_day) & (ratio == 1.0)
    for i in df.index[x_jump]:
        r = float(df.at[i, "close"] / prev_close[i])
        ratio[i] *= r
        used.append({"date": df.at[i, "date"], "ratio": r, "source": "x_gap", "implied_ratio": None,
                     "prev_close_table": None, "prev_close_panel": float(prev_close[i])})

    df["factor"] = (1.0 / ratio).cumprod()
    f = df["factor"]
    lim = np.where(df["date"] >= LIMIT_CHANGE_DATE, 0.10, 0.07)
    ref, close = df["ref"].to_numpy(), df["close"].to_numpy()
    tk = tick_size(close)
    with np.errstate(invalid="ignore"):
        up = (close >= ref * (1 + lim) - tk - 1e-9) & (close == df["high"].to_numpy())
        down = (close <= ref * (1 - lim) + tk + 1e-9) & (close == df["low"].to_numpy())
    df["limit_up"] = np.where(np.isnan(ref), 0.0, up.astype(float))
    df["limit_down"] = np.where(np.isnan(ref), 0.0, down.astype(float))
    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "open": df["open"] * f, "high": df["high"] * f, "low": df["low"] * f, "close": df["close"] * f,
        "volume": df["volume"] / f,  # Qlib 慣例：還原股數＝原始股數／factor，使 價×量 ≈ 成交金額
        "amount": df["amount"],
        "vwap": df["amount"] / df["volume"] * f,
        "factor": f,
        "limit_up": df["limit_up"], "limit_down": df["limit_down"],
    })
    out["change"] = out["close"] / out["close"].shift(1) - 1
    return out, used


# ---------------------------------------------------------------- 4/5. Qlib
def taiex_from_raw(dates) -> pd.DataFrame:
    """從 MI_INDEX 快取讀發行量加權股價指數收盤（主專案 taiex_daily.csv 沒有的日子用這個補）。"""
    rows = []
    for d in dates:
        p = RAW / "daily" / f"{d.year}" / f"{d:%Y%m%d}.json.gz"
        if not p.exists():
            continue
        for t in load_gz(p).get("tables") or []:
            if "價格指數" in str(t.get("title")):
                for r in t.get("data", []):
                    if str(r[0]).strip() == "發行量加權股價指數":
                        c = float(str(r[1]).replace(",", ""))
                        rows.append({"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 0.0, "turnover": 0.0})
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "turnover"])


def write_benchmark(calendar: pd.DatetimeIndex) -> dict:
    t = pd.read_csv(TAIEX_CSV, parse_dates=["date"]) if TAIEX_CSV.exists() else taiex_from_raw([])
    gap = calendar.difference(pd.DatetimeIndex(t["date"]))
    if len(gap):  # 主專案的指數檔沒有（或沒有那幾天）時，從日資料快取補
        t = pd.concat([t, taiex_from_raw(gap)], ignore_index=True).sort_values("date")
    t = t[(t["date"] >= calendar[0]) & (t["date"] <= calendar[-1])]
    info = {"bench_missing_days": int(len(calendar.difference(t["date"]))),
            "bench_extra_days": int(len(pd.DatetimeIndex(t["date"]).difference(calendar)))}
    t = t[t["date"].isin(calendar)]  # 不讓指數多出的日期混進交易日曆
    out = pd.DataFrame({
        "date": t["date"].dt.strftime("%Y-%m-%d"), "open": t["open"], "high": t["high"], "low": t["low"],
        "close": t["close"], "volume": t["volume"], "amount": t["turnover"], "vwap": t["close"],
        "factor": 1.0, "limit_up": 0.0, "limit_down": 0.0,
    })
    out["change"] = out["close"].pct_change()
    out.to_csv(CSV_DIR / f"{BENCHMARK}.csv", index=False)
    return info


def write_instruments(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
    inst_dir = QLIB_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    span = panel.groupby("code")["date"].agg(["min", "max"])
    with open(inst_dir / "all.txt", "w", encoding="utf-8", newline="\n") as f:
        for code, r in span.iterrows():
            f.write(f"{code}\t{r['min']:%Y-%m-%d}\t{r['max']:%Y-%m-%d}\n")
    with open(inst_dir / "index.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{BENCHMARK}\t{calendar[0]:%Y-%m-%d}\t{calendar[-1]:%Y-%m-%d}\n")

    # 每月前 N 名：以月底往回 POOL_LOOKBACK 個交易日平均成交金額排序（只用當時已知資料，次月生效）
    amt = panel.pivot(index="date", columns="code", values="amount").reindex(calendar)
    avg = amt.fillna(0).rolling(POOL_LOOKBACK, min_periods=POOL_LOOKBACK).mean()
    month_end = calendar.to_series().groupby(calendar.to_period("M")).max()
    months = list(month_end.index)
    info = {}
    for pool, top_n in POOLS.items():
        spans: dict[str, list[list[pd.Timestamp]]] = {}
        for k in range(len(months) - 1):
            rank_day = month_end.iloc[k]
            if pd.isna(avg.loc[rank_day]).all():
                continue
            top = avg.loc[rank_day].nlargest(top_n).index
            nxt = calendar[calendar.to_period("M") == months[k + 1]]
            s, e = nxt[0], nxt[-1]
            for code in top:
                # 只在該股仍上市期間生效
                e_code = min(e, span.at[code, "max"])
                if e_code < s:
                    continue
                lst = spans.setdefault(code, [])
                prev_idx = calendar.get_loc(lst[-1][1]) if lst else None
                if lst and prev_idx is not None and calendar.get_loc(s) == prev_idx + 1:
                    lst[-1][1] = e_code
                else:
                    lst.append([s, e_code])
        n_lines = 0
        with open(inst_dir / f"{pool}.txt", "w", encoding="utf-8", newline="\n") as f:
            for code in sorted(spans):
                for s, e in spans[code]:
                    f.write(f"{code}\t{s:%Y-%m-%d}\t{e:%Y-%m-%d}\n")
                    n_lines += 1
        info |= {f"{pool}_stocks_ever": len(spans), f"{pool}_lines": n_lines}
    return info


def run_dump() -> None:
    if QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)
    cmd = [sys.executable, str(ROOT / "scripts" / "dump_bin.py"), "dump_all",
           f"--data_path={CSV_DIR}", f"--qlib_dir={QLIB_DIR}", "--freq=day",
           "--date_field_name=date", "--exclude_fields=date,symbol", "--max_workers=8"]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- 6. 品質報告
def quality_report(panel: pd.DataFrame, csv_frames: dict[str, pd.DataFrame], used_events: pd.DataFrame,
                   pool_info: dict, calendar: pd.DatetimeIndex) -> dict:
    allp = pd.concat([d.assign(code=c) for c, d in csv_frames.items()], ignore_index=True)
    allp["date"] = pd.to_datetime(allp["date"])
    allp["year"] = allp["date"].dt.year
    span = panel.groupby("code")["date"].agg(["min", "max"])
    last = calendar[-1]
    ret = allp["change"]
    big = allp[ret.abs() > 0.105]
    by_year = allp.groupby("year").agg(stocks=("code", "nunique"), rows=("code", "size")).reset_index()
    tab = used_events[~used_events["source"].isin(["implied", "x_gap"])]
    agree = tab.dropna(subset=["implied_ratio"])
    rep = {
        "calendar": {"first": f"{calendar[0]:%Y-%m-%d}", "last": f"{last:%Y-%m-%d}", "trading_days": len(calendar)},
        "stocks_total": int(panel["code"].nunique()),
        "stocks_active_at_end": int((span["max"] == last).sum()),
        "stocks_delisted_or_stopped": int((span["max"] < last).sum()),
        "rows": int(len(panel)),
        "by_year": by_year.to_dict("records"),
        "missing": {
            "no_trade_rows_dropped": int(panel["close"].isna().sum()),  # 有列出但當日無成交
            "zero_volume_rows_after_filter": int((allp["volume"] <= 0).sum()),
            "nan_in_output": {c: int(allp[c].isna().sum()) for c in ["open", "high", "low", "close", "volume", "factor"]},
            "high_lt_low": int((allp["high"] < allp["low"] - 1e-9).sum()),
            "close_outside_hl": int(((allp["close"] > allp["high"] + 1e-6) | (allp["close"] < allp["low"] - 1e-6)).sum()),
        },
        "events": {
            "table_events_applied": int(len(tab)),
            "by_source": used_events["source"].value_counts().to_dict(),
            "table_vs_implied_ratio_agree_within_0.5pct": float(((agree["ratio"] / agree["implied_ratio"] - 1).abs() < 0.005).mean()) if len(agree) else None,
            "table_prev_close_mismatch_gt_1pct": int(((tab["prev_close_table"] / tab["prev_close_panel"] - 1).abs() > 0.01).sum()),
        },
        "adjusted_returns": {
            "abs_gt_10.5pct_rows": int(len(big)),
            "abs_gt_10.5pct_share": float(len(big) / ret.notna().sum()),
            "abs_gt_30pct_rows": int((ret.abs() > 0.3).sum()),
            "worst": big.reindex(big["change"].abs().sort_values(ascending=False).index)
                        .head(15)[["code", "date", "close", "change", "factor"]]
                        .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        },
        "limit_flags": {"limit_up_share": float(allp["limit_up"].mean()), "limit_down_share": float(allp["limit_down"].mean())},
        "pool": pool_info,
    }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-parse", action="store_true", help="沿用已存在的 panel.parquet")
    ap.add_argument("--implied-tol", type=float, default=0.02, help="漲跌價差反推事件的門檻")
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--pools-only", action="store_true", help="只依 panel.parquet 重寫 instruments（不動其他資料）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    panel_path = PROCESSED / "panel.parquet"
    if args.skip_parse and panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        print("1. 解析日資料")
        panel = build_panel()
        panel.to_parquet(panel_path, index=False)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    traded = panel.dropna(subset=["open", "high", "low", "close"])
    traded = traded[traded["volume"] > 0]
    if args.pools_only:
        print("instruments：", write_instruments(traded, calendar))
        return

    print("2. 載入事件")
    events = load_events()
    events.to_csv(PROCESSED / "events.csv", index=False, encoding="utf-8-sig")
    print(f"  事件 {len(events)} 筆：", events["source"].value_counts().to_dict())

    print("3. 計算 factor 並輸出 CSV")
    if CSV_DIR.exists():
        shutil.rmtree(CSV_DIR)
    CSV_DIR.mkdir(parents=True)
    ev_by_code = dict(tuple(events.groupby("code")))
    csv_frames, used_all = {}, []
    for code, df in traded.groupby("code"):
        out, used = compute_stock(df.reset_index(drop=True), ev_by_code.get(code, events.iloc[:0]), args.implied_tol)
        out.to_csv(CSV_DIR / f"{code}.csv", index=False)
        csv_frames[code] = out
        used_all += [dict(u, code=code) for u in used]
    used_events = pd.DataFrame(used_all)
    used_events.to_csv(PROCESSED / "events_applied.csv", index=False, encoding="utf-8-sig")
    bench_info = write_benchmark(calendar)
    print("  基準指數：", bench_info)

    if not args.skip_dump:
        print("4. dump_bin")
        run_dump()
    print("5. instruments")
    pool_info = write_instruments(traded, calendar) | bench_info

    print("6. 品質報告")
    rep = quality_report(panel, csv_frames, used_events, pool_info, calendar)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_year",)}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
