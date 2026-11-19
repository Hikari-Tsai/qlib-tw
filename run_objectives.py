"""股票池大小 × 訓練目標：H=20 標籤、shallow 參數。

- 股票池：tw100、tw300（每月依前 60 日平均成交金額取前 100／300 名）。
- 訓練目標：
  - reg：迴歸每日橫截面 rank 標準化後的報酬（先前的做法）。
  - rank：LambdaRank，相關度 = 每日報酬五分位（0～4），著重排序前段。
  - loser：二元分類「是否落在當日最差 20%」，分數取負號（越可能是輸家分數越低）。
- 指標（每日計算後取平均，報酬為之後 20 日報酬）：
  - Rank IC。
  - 最差 20% 價差：分數最低 20% 的平均報酬 − 股票池平均。越負代表越會認輸家。
  - 最佳 20% 價差：分數最高 20% 的平均報酬 − 股票池平均。
- 每個股票池只用驗證期「最差 20% 價差」挑一個目標，再用驗證期挑調倉間隔（5／20 日），
  跑「剔除最差 20%、其餘等權」的測試期回測，並和同股票池、同調倉間隔的等權持有比較。
  2021-11-05 起的保留集不評估。

用法（在 qlib_tw 內）：uv run python run_objectives.py
結果：outputs/objectives/results.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from ruamel.yaml import YAML

from run_exclude_losers import metrics, run_backtest
from run_experiments import HOLDOUT_START, PARAMS, TEST_START, VALID_START

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "objectives"
H = 20
POOLS = ["tw100", "tw300"]
OBJECTIVES = ["reg", "rank", "loser"]
VALID_END, TEST_END = "2018-12-31", "2021-11-04"


def evaluate(score: pd.Series, label: pd.Series) -> dict:
    df = pd.DataFrame({"s": score, "y": label}).dropna()
    rows = []
    for _, g in df.groupby(level="datetime"):
        if len(g) < 20:
            continue
        pct = g["s"].rank(pct=True, method="first")
        mu = g["y"].mean()
        rows.append((g["s"].corr(g["y"], method="spearman"),
                     g.loc[pct <= 0.2, "y"].mean() - mu, g.loc[pct > 0.8, "y"].mean() - mu))
    r = pd.DataFrame(rows, columns=["ric", "bot", "top"])
    return {"RankIC": float(r["ric"].mean()), "RankICIR": float(r["ric"].mean() / r["ric"].std()),
            "bottom20_spread": float(r["bot"].mean()), "top20_spread": float(r["top"].mean())}


def train(obj: str, X: pd.DataFrame, y_rank: pd.Series, y_raw: pd.Series, tr, va) -> lgb.Booster:
    base = {"verbosity": -1, "num_threads": 16, "seed": 42, **PARAMS["shallow"]}
    if obj == "reg":
        params, yt, yv = {**base, "objective": "regression", "metric": "l2"}, y_rank[tr], y_rank[va]
    elif obj == "loser":
        q = y_raw.groupby(level="datetime").rank(pct=True)
        lab = (q <= 0.2).astype(float)
        params, yt, yv = {**base, "objective": "binary", "metric": "binary_logloss"}, lab[tr], lab[va]
    else:  # rank
        q = y_raw.groupby(level="datetime").rank(pct=True)
        lab = np.minimum((q * 5).apply(np.ceil) - 1, 4).clip(lower=0)
        params = {**base, "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "label_gain": list(range(5))}
        yt, yv = lab[tr], lab[va]
    dtr = lgb.Dataset(X[tr], yt)
    dva = lgb.Dataset(X[va], yv, reference=dtr)
    if obj == "rank":  # 資料依日期排序，每天一組
        dtr.set_group(X[tr].groupby(level="datetime").size().to_numpy())
        dva.set_group(X[va].groupby(level="datetime").size().to_numpy())
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = YAML(typ="safe", pure=True).load((ROOT / "workflow_tw_alpha158_lgbm.yaml").read_text(encoding="utf-8"))
    qlib.init(**{**cfg["qlib_init"], "provider_uri": str(ROOT / "qlib_data"), "kernels": 3})
    hc = cfg["data_handler_config"]
    bt = cfg["port_analysis_config"]["backtest"]
    cal = D.calendar(start_time=hc["start_time"], end_time=HOLDOUT_START)
    seg_end = lambda b: cal[cal.searchsorted(pd.Timestamp(b)) - (H + 1) - 1]
    t0 = time.time()
    results: dict = {}

    for pool in POOLS:
        handler = Alpha158(
            instruments=pool, start_time=hc["start_time"], end_time=hc["end_time"],
            fit_start_time=hc["fit_start_time"], fit_end_time=hc["fit_end_time"],
            label=([f"Ref($close, -{H + 1})/Ref($close, -1) - 1"], ["LABEL0"]),
            learn_processors=[{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
        )
        learn = handler.fetch(col_set=["feature", "label"], data_key="learn")
        y_raw = handler.fetch(col_set="label", data_key="raw")["LABEL0"].reindex(learn.index)
        X, y_rank = learn["feature"].astype(np.float32), learn["label"]["LABEL0"]
        del learn, handler
        gc.collect()
        print(f"[{pool}] 資料載入 {time.time() - t0:.0f} 秒；特徵 {X.shape}", flush=True)
        dates = X.index.get_level_values("datetime")
        ok = y_raw.notna().to_numpy()
        tr = (dates <= seg_end(VALID_START)) & ok
        va = (dates >= pd.Timestamp(VALID_START)) & (dates <= seg_end(TEST_START)) & ok
        use = dates >= pd.Timestamp(VALID_START)
        te_eval = (dates >= pd.Timestamp(TEST_START)) & (dates <= seg_end(HOLDOUT_START))

        res, preds = {}, {}
        for obj in OBJECTIVES:
            t1 = time.time()
            model = train(obj, X, y_rank, y_raw, tr, va)
            s = pd.Series(model.predict(X[use], num_iteration=model.best_iteration), index=X.index[use])
            if obj == "loser":
                s = -s
            preds[obj] = s
            s.to_pickle(OUT / f"pred_{pool}_{obj}.pkl")
            v = evaluate(s[va[use]], y_raw[va])
            t = evaluate(s[te_eval[use]], y_raw[te_eval])
            res[obj] = {"best_iter": int(model.best_iteration), "valid": v, "test": t}
            print(f"[{pool}] {obj:6s} iter={model.best_iteration:4d}  "
                  f"valid RankIC={v['RankIC']:.4f} bot={v['bottom20_spread']:+.4f} top={v['top20_spread']:+.4f} | "
                  f"test RankIC={t['RankIC']:.4f} bot={t['bottom20_spread']:+.4f} top={t['top20_spread']:+.4f}  "
                  f"({time.time() - t1:.0f}s)", flush=True)

        best = min(res, key=lambda o: res[o]["valid"]["bottom20_spread"])
        ex = {"exchange": {**bt["exchange_kwargs"]["exchange"],
                           "kwargs": {**bt["exchange_kwargs"]["exchange"]["kwargs"], "codes": pool}}}
        btp = {**bt, "exchange_kwargs": ex}
        val_bt = {}
        for reb in (5, 20):
            for exc in (0.0, 0.2):
                val_bt[f"exc{int(exc * 100)}_reb{reb}"] = metrics(
                    run_backtest(preds[best], exc, reb, VALID_START, VALID_END, btp))
        reb = max((5, 20), key=lambda r: val_bt[f"exc20_reb{r}"]["excess_with_cost"]["information_ratio"])
        test_bt = {f"exc{int(exc * 100)}_reb{reb}": metrics(run_backtest(preds[best], exc, reb, TEST_START, TEST_END, btp))
                   for exc in (0.2, 0.0)}
        for split, d in [("valid", val_bt), ("test", test_bt)]:
            for k, m in d.items():
                print(f"[{pool}] {split} {best} {k:12s} 累積 {m['cum_return_with_cost']:+.3f} 指數 {m['cum_bench']:+.3f} "
                      f"超額年化 {m['excess_with_cost']['annualized_return']:+.3f} "
                      f"IR {m['excess_with_cost']['information_ratio']:+.2f} 換手 {m['avg_daily_turnover']:.3f}", flush=True)
        results[pool] = {"models": res, "chosen_objective": best, "chosen_rebalance": reb,
                         "valid_backtest": val_bt, "test_backtest": test_bt}
        del X, y_rank, y_raw, preds
        gc.collect()

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成，共 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
