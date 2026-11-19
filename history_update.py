"""把 Qlib 資料中「歷史檔還沒有的交易日」的模型分數與股價、籌碼，附加到歷史檔（只新增、不改寫）。

模型版本列在 models/versions.json，每個版本一個分數檔；股價與指數各版本共用。
歷史檔（預設 state/history/，可用 QTW_HISTORY 覆寫）：
- scores_<版本>.csv：date, code, H1, H5, ...（當天股票池每檔各週期分數）
- prices.csv：date, code, open, close（未還原）, ret（還原後日報酬）, foreign, trust, dealer（買賣超股數）, margin, short（餘額張數）
  只記錄「近 62 個交易日（含當天）曾在股票池」的股票（持有期最長 60 日，算組合報酬用得到）
- twii.csv：date, close
- names.csv：code, name（最新名稱）

分數一旦寫入就不再重算：之後換新版模型，過去的排名與成績也不會被改寫。
新增的模型版本第一次執行時，會從 2019-01 回填（需要本機完整資料）。

用法：
    uv run python history_update.py                    # 用本機完整資料
    QTW_QLIB=work/qlib_data QTW_PROCESSED=work/processed uv run python history_update.py   # 每日：只補新的交易日
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
QLIB = Path(os.environ.get("QTW_QLIB") or ROOT / "qlib_data")
PROCESSED = Path(os.environ.get("QTW_PROCESSED") or ROOT / "data" / "processed")
HISTORY = Path(os.environ.get("QTW_HISTORY") or ROOT / "state" / "history")
MODELS = ROOT / "models"
START = "2019-01-01"
HOLD_LOOKBACK = 61  # 最長持有 60 日 + 買進前一天
MIN_LOOKBACK = 80   # 新日期前至少要有的交易日數（60 日特徵）


def read_csv(name: str) -> pd.DataFrame | None:
    p = HISTORY / name
    return pd.read_csv(p, dtype={"code": str}, parse_dates=["date"]) if p.exists() else None


def write_csv(df: pd.DataFrame, name: str) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY / (name + ".tmp")
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(tmp, index=False, float_format="%.6g")
    tmp.replace(HISTORY / name)


RAW = Path(os.environ.get("QTW_RAW") or ROOT / "data" / "raw")


def update_ref(code: str, cal: pd.DatetimeIndex) -> None:
    """從證交所日資料快取計算 ETF 的含息日報酬，附加到 ref_<代號>.csv。

    一般日：報酬 = 收盤 ÷ 參考價 − 1，參考價 = 收盤 − 漲跌價差（證交所的漲跌已扣除息、反映分割）。
    除權息日（漲跌標示 X）：參考價取除權息計算結果表的「除權息參考價」；查不到才用前一日收盤。
    """
    import gzip
    import re

    from build_dataset import roc_to_date

    name = f"ref_{code}.csv"
    old = read_csv(name)
    todo = [d for d in cal if d >= pd.Timestamp(START) and (old is None or d > old["date"].max())]
    if not todo:
        return
    files = sorted((RAW / "daily").glob("*/*.json.gz"))
    if old is None and (not files or files[0].name[:8] > "20190102"):
        print(f"{name} 不存在，且 {RAW} 沒有 2019 年起的原始資料，略過（請在本機用完整資料回填）")
        return
    refs = {}
    for p in (RAW / "events").glob("TWT49U_*.json.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            j = json.load(f)
        for r in j.get("data") or []:
            if str(r[1]).strip() == code:
                refs[roc_to_date(r[0])] = float(str(r[4]).replace(",", ""))
    prev = float(old["close"].iloc[-1]) if old is not None else None
    rows = []
    for d in todo:
        p = RAW / "daily" / f"{d.year}" / f"{d:%Y%m%d}.json.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            j = json.load(f)
        rec = None
        for t in j.get("tables") or []:
            fields = t.get("fields") or []
            if "證券代號" in fields:
                rec = next((dict(zip(fields, r)) for r in t["data"] if r[0].strip() == code), None)
                break
        if rec is None:
            continue
        try:
            close = float(rec["收盤價"].replace(",", ""))
        except ValueError:
            continue  # 當天沒有成交
        sign = re.sub(r"<[^>]*>", "", rec["漲跌(+/-)"]).strip().upper()
        chg = float(rec["漲跌價差"].replace(",", "") or 0)
        if sign in ("+", "-"):
            base = close - (chg if sign == "+" else -chg)
        elif sign == "X":
            base = refs.get(d, prev)
        else:
            base = close  # 平盤
        ret = close / base - 1 if base and prev is not None else None
        rows.append({"date": d, "close": close, "ret": None if ret is None else round(ret, 6)})
        prev = close
    if rows:
        new = pd.DataFrame(rows)
        write_csv(pd.concat([old, new]) if old is not None else new, name)
        print(f"{name}：+{len(new)} 列", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tw100")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    import qlib
    from qlib.data import D

    from tw_qlib.handlers import Alpha158Flow

    qlib.init(provider_uri=str(QLIB), region="tw", kernels=3)
    cal = pd.DatetimeIndex(D.calendar())
    versions = [v["id"] for v in json.loads((MODELS / "versions.json").read_text(encoding="utf-8"))["versions"]]

    def pending(last) -> list:
        return [d for d in cal if d >= pd.Timestamp(START) and (last is None or d > last)]

    def check_lookback(ds: list) -> None:
        if ds and cal.get_loc(ds[0]) < MIN_LOOKBACK:
            raise SystemExit(f"{ds[0]:%Y-%m-%d} 之前只有 {cal.get_loc(ds[0])} 個交易日的資料，不足以計算特徵；"
                             "新版本的回填請在本機用完整資料執行")

    # 1. 各版本的分數
    all_scores = []
    for v in versions:
        old = read_csv(f"scores_{v}.csv")
        new_dates = pending(old["date"].max() if old is not None else None)
        if new_dates:
            check_lookback(new_dates)
            metas = sorted((json.loads(p.read_text(encoding="utf-8")) for p in (MODELS / v).glob(f"{args.pool}_H*.json")),
                           key=lambda m: m["horizon"])
            handler = Alpha158Flow(instruments=args.pool, start_time=new_dates[0], end_time=new_dates[-1],
                                   fit_start_time=new_dates[0], fit_end_time=new_dates[-1], learn_processors=[])
            X = handler.fetch(col_set="feature").astype(np.float32)
            X = X[X.index.get_level_values("datetime").isin(new_dates)]
            sc = pd.DataFrame(index=X.index)
            for m in metas:
                model = lgb.Booster(model_file=str(MODELS / v / f"{args.pool}_H{m['horizon']}.txt"))
                sc[f"H{m['horizon']}"] = np.round(model.predict(X[m["features"]]), 6)
            sc = sc.reset_index().rename(columns={"datetime": "date", "instrument": "code"})
            old = pd.concat([old, sc]) if old is not None else sc
            write_csv(old, f"scores_{v}.csv")
            print(f"{v}：分數 +{len(sc)} 列（{new_dates[0]:%Y-%m-%d} ~ {new_dates[-1]:%Y-%m-%d}）", flush=True)
        else:
            print(f"{v}：分數已是最新", flush=True)
        all_scores.append(old[["date", "code"]])
    members = pd.concat(all_scores).drop_duplicates()

    # 2. 股價與籌碼（只補 prices.csv 還沒有的日期）
    old_prices, old_twii = read_csv("prices.csv"), read_csv("twii.csv")
    new_dates = pending(old_prices["date"].max() if old_prices is not None else None)
    if new_dates:
        check_lookback(new_dates)
        first_pos = cal.get_loc(new_dates[0])
        window = cal[max(0, first_pos - HOLD_LOOKBACK):cal.get_loc(new_dates[-1]) + 1]
        mem = pd.crosstab(members["date"], members["code"]).reindex(window, fill_value=0).astype(bool)
        keep = mem.rolling(HOLD_LOOKBACK + 1, min_periods=1).max().astype(bool).stack()
        keep = keep[keep].index
        codes = sorted(keep.get_level_values(1).unique())
        fields = {"open": "$open/$factor", "close": "$close/$factor", "ret": "$close/Ref($close, 1) - 1",
                  "foreign": "$foreign_net*$factor", "trust": "$trust_net*$factor", "dealer": "$dealer_net*$factor",
                  "margin": "$margin_bal*$factor/1000", "short": "$short_bal*$factor/1000"}
        pr = D.features(codes, list(fields.values()), start_time=new_dates[0], end_time=new_dates[-1])
        pr.columns = list(fields)
        pr = pr.reset_index().rename(columns={"datetime": "date", "instrument": "code"}).dropna(subset=["close"])
        pr = pr[pd.MultiIndex.from_frame(pr[["date", "code"]]).isin(keep)]
        for c in ("open", "close"):
            pr[c] = pr[c].round(2)
        for c in ("foreign", "trust", "dealer", "margin", "short"):
            pr[c] = pr[c].round(0)
        tw = D.features(["TWII"], ["$close"], start_time=new_dates[0], end_time=new_dates[-1])
        tw = tw.reset_index().rename(columns={"datetime": "date", "$close": "close"})[["date", "close"]].dropna()
        write_csv(pd.concat([old_prices, pr]) if old_prices is not None else pr, "prices.csv")
        write_csv(pd.concat([old_twii, tw]) if old_twii is not None else tw, "twii.csv")
        print(f"價格 +{len(pr)} 列、指數 +{len(tw)} 列", flush=True)

    # 3. 參考指標：0050（元大台灣50，含息、已處理分割）
    update_ref("0050", cal)

    # 4. 名稱
    panel = pd.read_parquet(PROCESSED / "panel.parquet", columns=["date", "code", "name"])
    names = panel.sort_values("date").drop_duplicates("code", keep="last")[["code", "name"]]
    names["name"] = names["name"].str.strip()
    p = HISTORY / "names.csv"
    if p.exists():
        prev = pd.read_csv(p, dtype=str)
        names = pd.concat([prev[~prev["code"].isin(names["code"])], names])
    names.sort_values("code").to_csv(p, index=False)
    print(f"歷史檔：{HISTORY}")


if __name__ == "__main__":
    main()
