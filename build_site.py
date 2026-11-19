"""從歷史檔（history_update.py 產生）建出靜態網頁資料：site/data.js 與 site/api/。不需要 Qlib。

每個模型版本（models/versions.json）各自計算排名與組合績效；股價、指數、股票池等權共用。

組合績效（每個週期 H、每個 N；不計手續費）：
- 第 d 天收盤後排名；第 d+1 天收盤以等權買進名單，持有 H 個交易日（賺 d+2 ~ d+1+H 的報酬），名單內不再平衡。
- 每天建立一份新名單，組合每天平均分配給仍在持有期內的 H 份名單。
- 做多 TopN、LowN（網頁上以做空呈現）、多空（TopN − LowN）；對照：股票池等權（前一天成員、每日等權）、加權指數（不含息）。
- 停牌日報酬記 0；不考慮漲跌停無法成交。

用法（在 qlib_tw 內）：uv run python build_site.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HISTORY = Path(os.environ.get("QTW_HISTORY") or ROOT / "state" / "history")
SITE = Path(os.environ.get("QTW_SITE") or ROOT / "site")
MODELS = ROOT / "models"
POOL = "tw100"
NS = [5, 10, 20]
API_RANK_DAYS = 120
API_PRICE_DAYS = 250


def portfolios(sc: pd.DataFrame, hcols: list[str], cal: pd.DatetimeIndex, Gv: np.ndarray, col: dict) -> dict:
    dates = list(cal)
    pos = {d: i for i, d in enumerate(dates)}
    out = {}
    for hc in hcols:
        h = int(hc[1:])
        by_day = {d: g.droplevel(0).sort_values(ascending=False) for d, g in sc[hc].groupby(level="date")}
        for n in NS:
            acc = {"top": np.zeros(len(dates)), "low": np.zeros(len(dates))}
            count = np.zeros(len(dates))
            for d, s in by_day.items():
                if len(s) < 2 * n or d not in pos:
                    continue
                e = pos[d] + 1
                if e >= len(dates) - 1:
                    continue
                end = min(e + h, len(dates) - 1)
                for side, picks in (("top", s.index[:n]), ("low", s.index[-n:])):
                    idx = [col[c] for c in picks if c in col]
                    if idx:
                        port = (Gv[e:end + 1, idx] / Gv[e, idx]).mean(axis=1)
                        acc[side][e + 1:end + 1] += port[1:] / port[:-1] - 1
                count[e + 1:end + 1] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                top = pd.Series(np.where(count > 0, acc["top"] / count, np.nan), index=cal)
                low = pd.Series(np.where(count > 0, acc["low"] / count, np.nan), index=cal)
            out[f"H{h}_N{n}_long"], out[f"H{h}_N{n}_low"], out[f"H{h}_N{n}_ls"] = top, low, top - low
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    vcfg = json.loads((MODELS / "versions.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(HISTORY / "prices.csv", dtype={"code": str}, parse_dates=["date"])
    twii = pd.read_csv(HISTORY / "twii.csv", parse_dates=["date"]).set_index("date")["close"]
    names = pd.read_csv(HISTORY / "names.csv", dtype=str).set_index("code")["name"].fillna("")
    scores = {v["id"]: pd.read_csv(HISTORY / f"scores_{v['id']}.csv", dtype={"code": str}, parse_dates=["date"])
              for v in vcfg["versions"] if (HISTORY / f"scores_{v['id']}.csv").exists()}
    vids = list(scores)

    cal = pd.DatetimeIndex(sorted(set().union(*(set(s["date"]) for s in scores.values()))))
    last = cal[-1]
    R = prices.pivot(index="date", columns="code", values="ret").reindex(cal).fillna(0.0)
    Gv = (1 + R).cumprod().to_numpy()
    col = {c: j for j, c in enumerate(R.columns)}
    members = scores[vids[0]].groupby("date")["code"].apply(list)  # 股票池與模型版本無關
    pool_ret = pd.Series(np.nan, index=cal)
    for i in range(1, len(cal)):
        mem = [c for c in members.get(cal[i - 1], []) if c in col]
        if mem:
            pool_ret.iloc[i] = R.iloc[i][mem].mean()
    twii_ret = twii.reindex(cal).pct_change(fill_method=None)
    ref_path = HISTORY / "ref_0050.csv"
    etf_ret = (pd.read_csv(ref_path, parse_dates=["date"]).set_index("date")["ret"].reindex(cal)
               if ref_path.exists() else pd.Series(np.nan, index=cal))
    px_last = prices[prices["date"] == last].set_index("code")
    fnum = lambda x, d=2: None if x is None or pd.isna(x) else round(float(x), d)
    daily_list = lambda s: [None if pd.isna(x) else round(float(x), 6) for x in s]

    data = {"pool": POOL, "asOf": f"{last:%Y-%m-%d}",
            "generated": dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M"),
            "ns": NS, "dates": [f"{d:%Y-%m-%d}" for d in cal], "defaultVersion": vcfg["default"],
            "common": {"pool": daily_list(pool_ret), "twii": daily_list(twii_ret), "etf0050": daily_list(etf_ret)},
            "versions": [], "byVersion": {}}
    api_v = {}
    for v in vcfg["versions"]:
        vid = v["id"]
        if vid not in scores:
            continue
        s = scores[vid]
        hcols = sorted((c for c in s.columns if c.startswith("H")), key=lambda c: int(c[1:]))
        horizons = [int(c[1:]) for c in hcols]
        metas = {m["horizon"]: m for m in (json.loads(p.read_text(encoding="utf-8")) for p in (MODELS / vid).glob(f"{POOL}_H*.json"))}
        sc = s.set_index(["date", "code"])
        series = portfolios(sc, hcols, cal, Gv, col)
        rank_tbl = {hc: sc[hc].groupby(level="date").rank(ascending=False, method="min") for hc in hcols}
        latest = sc.xs(last, level="date") if last in sc.index.get_level_values("date") else sc.iloc[:0].droplevel(0)
        ranking = [{"code": code, "name": names.get(code, ""), "open": fnum(px_last["open"].get(code)),
                    "close": fnum(px_last["close"].get(code)),
                    **{f"s{h}": round(float(r[hc]), 5) for h, hc in zip(horizons, hcols)},
                    **{f"r{h}": int(rank_tbl[hc].loc[(last, code)]) for h, hc in zip(horizons, hcols)}}
                   for code, r in latest.iterrows()]
        vdates = pd.DatetimeIndex(sorted(s["date"].unique()))
        rank_ref = {}
        for h, hc in zip(horizons, hcols):
            ref = {}
            for name, back in (("d1", 1), ("dh", h)):
                if len(vdates) > back:
                    d = vdates[-1 - back]
                    ref[name] = {"date": f"{d:%Y-%m-%d}", "ranks": {c: int(x) for c, x in rank_tbl[hc].xs(d, level="date").items()}}
            rank_ref[str(h)] = ref
        evals = {h: m.get("test_metrics") or m.get("eval_metrics") for h, m in metas.items()}
        data["versions"].append({k: v[k] for k in v})
        data["byVersion"][vid] = {
            "horizons": horizons,
            "featureSets": {str(h): metas[h]["feature_set"] for h in horizons if h in metas},
            "evalRankIC": {str(h): round(evals[h]["RankIC"], 4) for h in horizons if h in metas and evals[h]},
            "evalName": v.get("evalName", ""),
            "daily": {k: daily_list(x) for k, x in series.items()},
            "ranking": ranking, "rankRef": rank_ref,
        }
        api_v[vid] = (horizons, hcols, rank_tbl, latest, metas)

    SITE.mkdir(parents=True, exist_ok=True)
    js = "window.QDATA = " + json.dumps(data, ensure_ascii=False, allow_nan=False) + ";\n"
    (SITE / "data.js").write_text(js, encoding="utf-8")

    # 靜態 JSON API
    api = SITE / "api"
    (api / "stocks").mkdir(parents=True, exist_ok=True)
    rdates, pdates = cal[-API_RANK_DAYS:], cal[-API_PRICE_DAYS:]
    codes = sorted(set().union(*(set(scores[v].loc[scores[v]["date"].isin(rdates), "code"]) for v in api_v)))
    pr = prices[prices["date"].isin(pdates)].set_index(["code", "date"]).sort_index()
    pr_codes = set(pr.index.get_level_values("code"))
    for old in (api / "stocks").glob("*.json"):
        if old.stem not in codes:
            old.unlink()
    latest_codes = set().union(*(set(x[3].index) for x in api_v.values()))
    index_rows = []
    for code in codes:
        f = pr.xs(code, level="code").reindex(pdates) if code in pr_codes else pd.DataFrame(index=pdates)
        colv = lambda c, d: [fnum(x, d) for x in (f[c] if c in f else pd.Series(np.nan, index=pdates))]
        by_v_latest, by_v_hist = {}, {}
        for vid, (horizons, hcols, rank_tbl, latest, _) in api_v.items():
            by_v_latest[vid] = ({f"H{h}": {"rank": int(rank_tbl[hc].loc[(last, code)]), "score": round(float(latest.loc[code, hc]), 5)}
                                 for h, hc in zip(horizons, hcols)} if code in latest.index else None)
            by_v_hist[vid] = {f"H{h}": ([None if pd.isna(x) else int(x) for x in rank_tbl[hc].xs(code, level="code").reindex(rdates)]
                                        if code in rank_tbl[hc].index.get_level_values("code") else [None] * len(rdates))
                              for h, hc in zip(horizons, hcols)}
        doc = {"code": code, "name": names.get(code, ""), "asOf": f"{last:%Y-%m-%d}", "pool": POOL,
               "inPool": code in latest_codes, "poolSize": len(latest_codes),
               "latest": by_v_latest, "rankHistory": {"dates": [f"{d:%Y-%m-%d}" for d in rdates], **by_v_hist},
               "daily": {"dates": [f"{d:%Y-%m-%d}" for d in pdates], "open": colv("open", 2), "close": colv("close", 2),
                         "foreign_net": colv("foreign", 0), "trust_net": colv("trust", 0), "dealer_net": colv("dealer", 0),
                         "margin_lots": colv("margin", 0), "short_lots": colv("short", 0)},
               "units": {"open/close": "元（未還原開盤／收盤價）", "foreign_net/trust_net/dealer_net": "股（買賣超）",
                         "margin_lots/short_lots": "張（融資／融券餘額）", "rank": "1 = 當天股票池內分數最高；依模型版本分開"}}
        (api / "stocks" / f"{code}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        index_rows.append({"code": code, "name": names.get(code, ""), "inPool": code in latest_codes})
    (api / "index.json").write_text(json.dumps({
        "asOf": f"{last:%Y-%m-%d}", "pool": POOL, "defaultVersion": vcfg["default"], "versions": vcfg["versions"],
        "models": {vid: {f"H{h}": {"featureSet": m["feature_set"], "train": m["train"], "valid": m["valid"]}
                         for h, m in x[4].items()} for vid, x in api_v.items()},
        "stocks": index_rows,
        "endpoints": {"latest": "api/latest.json", "stock": "api/stocks/{code}.json"},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (api / "latest.json").write_text(json.dumps({
        "asOf": f"{last:%Y-%m-%d}", "pool": POOL,
        "versions": {vid: [{"code": r["code"], "name": r["name"], "open": r["open"], "close": r["close"],
                            **{f"H{h}": {"rank": r[f"r{h}"], "score": r[f"s{h}"]} for h in data["byVersion"][vid]["horizons"]}}
                           for r in data["byVersion"][vid]["ranking"]] for vid in api_v},
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"網頁資料 {last:%Y-%m-%d}：版本 {vids}，data.js {len(js) / 1e6:.1f} MB、API {len(codes)} 檔")


if __name__ == "__main__":
    main()
