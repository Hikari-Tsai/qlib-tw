"""加入三大法人與融資融券特徵（tw_qlib.handlers.Alpha158Flow）是否帶來增量資訊。

- 法人資料從 2012-05 才有，所以兩組模型都改用 2012-09-01 起的資料訓練（到 2016 年底，交界往前留 H+1 天）；
  驗證 2017–2018、測試 2019-01 ~ 2021-11-04，保留集不評估。
- base：只用 Alpha158；flow：Alpha158 + 法人／融資融券 18 個特徵。H=20 標籤、迴歸、shallow 參數。
- 指標同 run_objectives.py（Rank IC、最差／最佳 20% 價差）。
- 回測：flow 模型在驗證期比較 TopkDropout(20,3)、剔除最差 20%（每 5／20 日調倉），
  挑扣成本後超額 IR 最高者跑測試期，並和同股票池等權持有（每 5 日調倉）比較。

--wait：等 crawl_flows.py 把 2012-05-02 ~ 2021-11-04 爬完，再執行 build_flows.py 和實驗。

用法（在 qlib_tw 內）：uv run python run_flows_exp.py [--wait]
結果：outputs/flows/results.json
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "flows"
H = 20
POOLS = ["tw100", "tw300"]
TRAIN_START, VALID_START, TEST_START, HOLDOUT_START = "2012-09-01", "2017-01-01", "2019-01-01", "2021-11-05"
VALID_END, TEST_END = "2018-12-31", "2021-11-04"


def wait_for_crawl() -> None:
    cal = pd.read_csv(ROOT / "qlib_data" / "calendars" / "day.txt", header=None)[0]
    days = [pd.Timestamp(d) for d in cal if "2012-05-02" <= d <= TEST_END]
    while True:
        missing = sum(not (ROOT / "data" / "raw" / n / f"{d.year}" / f"{d:%Y%m%d}.json.gz").exists()
                      for n in ("t86", "margin") for d in days)
        if missing == 0:
            return
        print(f"{time.strftime('%H:%M')} 還差 {missing} 個快取檔", flush=True)
        time.sleep(600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.wait:
        wait_for_crawl()
        subprocess.run([sys.executable, str(ROOT / "build_flows.py")], check=True)

    import qlib
    from qlib.data import D
    from ruamel.yaml import YAML

    from run_exclude_losers import metrics, run_backtest
    from run_experiments import PARAMS
    from run_objectives import evaluate
    from tw_qlib.handlers import FLOW_FEATURES, Alpha158Flow

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = YAML(typ="safe", pure=True).load((ROOT / "workflow_tw_alpha158_lgbm.yaml").read_text(encoding="utf-8"))
    qlib.init(**{**cfg["qlib_init"], "provider_uri": str(ROOT / "qlib_data"), "kernels": 3})
    bt = cfg["port_analysis_config"]["backtest"]
    cal = D.calendar(start_time="2008-01-01", end_time=HOLDOUT_START)
    seg_end = lambda b: cal[cal.searchsorted(pd.Timestamp(b)) - (H + 1) - 1]
    t0 = time.time()
    results: dict = {}

    for pool in POOLS:
        handler = Alpha158Flow(
            instruments=pool, start_time="2012-05-01", end_time=TEST_END,
            fit_start_time=TRAIN_START, fit_end_time="2016-12-31",
            label=([f"Ref($close, -{H + 1})/Ref($close, -1) - 1"], ["LABEL0"]),
            learn_processors=[{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
        )
        learn = handler.fetch(col_set=["feature", "label"], data_key="learn")
        y_raw = handler.fetch(col_set="label", data_key="raw")["LABEL0"].reindex(learn.index)
        X, y = learn["feature"].astype(np.float32), learn["label"]["LABEL0"]
        del learn, handler
        gc.collect()
        X = X[X.index.get_level_values("datetime") >= pd.Timestamp(TRAIN_START)]
        y, y_raw = y.reindex(X.index), y_raw.reindex(X.index)
        print(f"[{pool}] 資料載入 {time.time() - t0:.0f} 秒；特徵 {X.shape}；"
              f"法人特徵缺值比例 {X[list(FLOW_FEATURES)].isna().mean().mean():.3f}", flush=True)
        dates = X.index.get_level_values("datetime")
        ok = y_raw.notna().to_numpy()
        tr = (dates <= seg_end(VALID_START)) & ok
        va = (dates >= pd.Timestamp(VALID_START)) & (dates <= seg_end(TEST_START)) & ok
        use = dates >= pd.Timestamp(VALID_START)
        te_eval = (dates >= pd.Timestamp(TEST_START)) & (dates <= seg_end(HOLDOUT_START))

        res, preds = {}, {}
        for name, cols in [("base", [c for c in X.columns if c not in FLOW_FEATURES]), ("flow", list(X.columns))]:
            dtr = lgb.Dataset(X.loc[tr, cols], y[tr])
            dva = lgb.Dataset(X.loc[va, cols], y[va], reference=dtr)
            model = lgb.train({"objective": "regression", "metric": "l2", "verbosity": -1, "num_threads": 16,
                               "seed": 42, **PARAMS["shallow"]}, dtr, num_boost_round=2000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            s = pd.Series(model.predict(X.loc[use, cols], num_iteration=model.best_iteration), index=X.index[use])
            s.to_pickle(OUT / f"pred_{pool}_{name}.pkl")
            preds[name] = s
            v, t = evaluate(s[va[use]], y_raw[va]), evaluate(s[te_eval[use]], y_raw[te_eval])
            imp = pd.Series(model.feature_importance("gain"), index=cols)
            res[name] = {"best_iter": int(model.best_iteration), "valid": v, "test": t,
                         "flow_gain_share": float(imp[imp.index.isin(list(FLOW_FEATURES))].sum() / imp.sum()),
                         "top_features": {k: round(float(x / imp.sum()), 4) for k, x in imp.nlargest(15).items()}}
            print(f"[{pool}] {name:5s} iter={model.best_iteration:4d}  "
                  f"valid RankIC={v['RankIC']:.4f} bot={v['bottom20_spread']:+.4f} top={v['top20_spread']:+.4f} | "
                  f"test RankIC={t['RankIC']:.4f} bot={t['bottom20_spread']:+.4f} top={t['top20_spread']:+.4f}  "
                  f"法人特徵 gain 占比 {res[name]['flow_gain_share']:.2f}", flush=True)
        print(f"[{pool}] flow 前 15 重要特徵：{res['flow']['top_features']}", flush=True)

        ex = {"exchange": {**bt["exchange_kwargs"]["exchange"],
                           "kwargs": {**bt["exchange_kwargs"]["exchange"]["kwargs"], "codes": pool}}}
        btp = {**bt, "exchange_kwargs": ex}
        val_bt = {"ew_reb5": metrics(run_backtest(preds["flow"], 0.0, 5, VALID_START, VALID_END, btp)),
                  "topk20": metrics(run_topk(preds["flow"], VALID_START, VALID_END, btp))}
        for reb in (5, 20):
            val_bt[f"exc20_reb{reb}"] = metrics(run_backtest(preds["flow"], 0.2, reb, VALID_START, VALID_END, btp))
        best = max((k for k in val_bt if k != "ew_reb5"),
                   key=lambda k: val_bt[k]["excess_with_cost"]["information_ratio"])
        test_bt = {"ew_reb5": metrics(run_backtest(preds["flow"], 0.0, 5, TEST_START, TEST_END, btp))}
        test_bt[best] = metrics(run_topk(preds["flow"], TEST_START, TEST_END, btp) if best == "topk20" else
                                run_backtest(preds["flow"], 0.2, int(best.split("reb")[1]), TEST_START, TEST_END, btp))
        for split, d in [("valid", val_bt), ("test", test_bt)]:
            for k, m in d.items():
                print(f"[{pool}] {split} flow {k:12s} 累積 {m['cum_return_with_cost']:+.3f} 指數 {m['cum_bench']:+.3f} "
                      f"超額年化 {m['excess_with_cost']['annualized_return']:+.3f} "
                      f"IR {m['excess_with_cost']['information_ratio']:+.2f} 換手 {m['avg_daily_turnover']:.3f}", flush=True)
        results[pool] = {"models": res, "chosen_strategy": best, "valid_backtest": val_bt, "test_backtest": test_bt}
        del X, y, y_raw, preds
        gc.collect()

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成，共 {time.time() - t0:.0f} 秒")


def run_topk(pred: pd.Series, start: str, end: str, bt: dict) -> pd.DataFrame:
    import copy

    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy

    ex = copy.deepcopy(bt["exchange_kwargs"])
    ex["exchange"]["kwargs"].update(start_time=start, end_time=end)
    report, _ = backtest_daily(start_time=start, end_time=end,
                               strategy=TopkDropoutStrategy(signal=pred, topk=20, n_drop=3),
                               account=bt["account"], benchmark=bt["benchmark"], exchange_kwargs=ex)
    return report


if __name__ == "__main__":
    main()
