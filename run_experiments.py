"""Alpha158 + LightGBM：標籤期間 × 模型參數 的實驗網格。

- 標籤：在 t+1 收盤買進、持有 H 天的報酬 Ref($close,-(H+1))/Ref($close,-1)-1，H ∈ {1, 5, 20}，
  每日橫截面 rank 標準化後當作訓練目標。
- 切分與設定檔相同（訓練 2008–2016、驗證 2017–2018、測試 2019-01 ~ 2021-11-04），
  各段交界往前刪掉 H+1 個交易日，避免標籤跨段；2021-11-05 起的保留集不評估。
- 只用驗證集 Rank IC 挑每個 H 的最佳參數，再對它跑測試期回測（與設定檔相同的成本、漲跌停、TopkDropout）。

用法（在 qlib_tw 內）：uv run python run_experiments.py
結果：outputs/experiments/results.json、各組 pred_*.pkl、回測 report_*.pkl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "experiments"

HORIZONS = [1, 5, 20]
VALID_START, TEST_START, HOLDOUT_START = "2017-01-01", "2019-01-01", "2021-11-05"
PARAMS = {
    # Qlib 為中證 300 調好的參數（設定檔原樣）
    "qlib_csi300": dict(learning_rate=0.2, num_leaves=210, max_depth=8, colsample_bytree=0.8879,
                        subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768),
    # 中等：較低學習率、較少葉子、正則化放寬
    "moderate": dict(learning_rate=0.05, num_leaves=64, max_depth=-1, min_data_in_leaf=200,
                     colsample_bytree=0.8, subsample=0.8, subsample_freq=1, lambda_l2=10.0),
    # 保守：淺樹、大葉節點，降低對雜訊的擬合
    "shallow": dict(learning_rate=0.02, num_leaves=31, max_depth=6, min_data_in_leaf=1000,
                    colsample_bytree=0.7, subsample=0.7, subsample_freq=1, lambda_l2=50.0),
}


def daily_ic(pred: pd.Series, label: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    g = df.groupby(level="datetime")
    ic = g.apply(lambda x: x["p"].corr(x["y"]))
    ric = g.apply(lambda x: x["p"].corr(x["y"], method="spearman"))
    return ic, ric


def summarize(ic: pd.Series, ric: pd.Series) -> dict:
    return {"IC": float(ic.mean()), "ICIR": float(ic.mean() / ic.std()),
            "RankIC": float(ric.mean()), "RankICIR": float(ric.mean() / ric.std()),
            "RankIC_by_year": {int(y): round(float(v), 4) for y, v in ric.groupby(ric.index.year).mean().items()}}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = YAML(typ="safe", pure=True).load((ROOT / "workflow_tw_alpha158_lgbm.yaml").read_text(encoding="utf-8"))
    # 子行程少一點：本機記憶體被其他程式占用時，6 個子行程會 MemoryError
    qlib.init(**{**cfg["qlib_init"], "provider_uri": str(ROOT / "qlib_data"), "kernels": 3})
    hc = cfg["data_handler_config"]

    t0 = time.time()
    exprs = [f"Ref($close, -{h + 1})/Ref($close, -1) - 1" for h in HORIZONS]
    names = [f"LABEL{h}" for h in HORIZONS]
    handler = Alpha158(
        instruments=hc["instruments"], start_time=hc["start_time"], end_time=hc["end_time"],
        fit_start_time=hc["fit_start_time"], fit_end_time=hc["fit_end_time"],
        label=(exprs, names),
        learn_processors=[{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
    )
    learn = handler.fetch(col_set=["feature", "label"], data_key="learn")
    raw_label = handler.fetch(col_set="label", data_key="raw")
    X_all = learn["feature"].astype(np.float32)
    print(f"資料載入 {time.time() - t0:.0f} 秒；特徵 {X_all.shape}", flush=True)

    cal = D.calendar(start_time=hc["start_time"], end_time=HOLDOUT_START)
    dates = X_all.index.get_level_values("datetime")

    def seg_end(boundary: str, h: int) -> pd.Timestamp:
        i = cal.searchsorted(pd.Timestamp(boundary))  # 該段之後第一個交易日的位置
        return cal[i - (h + 1) - 1]  # 往前留 H+1 天，讓標籤不跨段

    results, preds = {}, {}
    for h in HORIZONS:
        y_all = learn["label"][f"LABEL{h}"]
        y_raw = raw_label[f"LABEL{h}"]
        tr = (dates <= seg_end(VALID_START, h))
        va = (dates >= pd.Timestamp(VALID_START)) & (dates <= seg_end(TEST_START, h))
        te_pred = (dates >= pd.Timestamp(TEST_START))  # 測試期全部預測（只用到過去特徵）
        te_eval = te_pred & (dates <= seg_end(HOLDOUT_START, h))  # IC 只算到保留集前 H+1 天
        tr_m, va_m = tr & y_all.notna().to_numpy(), va & y_all.notna().to_numpy()
        dtr = lgb.Dataset(X_all[tr_m], y_all[tr_m])
        dva = lgb.Dataset(X_all[va_m], y_all[va_m], reference=dtr)
        for pname, p in PARAMS.items():
            key = f"H{h}_{pname}"
            t1 = time.time()
            model = lgb.train({"objective": "regression", "metric": "l2", "verbosity": -1, "num_threads": 16,
                               "seed": 42, **p}, dtr, num_boost_round=2000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            pv = pd.Series(model.predict(X_all[va], num_iteration=model.best_iteration), index=X_all.index[va])
            pt = pd.Series(model.predict(X_all[te_pred], num_iteration=model.best_iteration), index=X_all.index[te_pred])
            v = summarize(*daily_ic(pv, y_raw[va]))
            t = summarize(*daily_ic(pt[te_eval[te_pred]], y_raw[te_eval]))
            results[key] = {"horizon": h, "params": pname, "best_iter": int(model.best_iteration),
                            "valid": v, "test": t}
            preds[key] = pt
            pt.to_pickle(OUT / f"pred_{key}.pkl")
            print(f"{key:20s} iter={model.best_iteration:4d}  valid RankIC={v['RankIC']:.4f}  "
                  f"test RankIC={t['RankIC']:.4f}  ({time.time() - t1:.0f}s)", flush=True)

    # 每個 H 依驗證集 Rank IC 選最佳參數，跑測試期回測
    bt = cfg["port_analysis_config"]["backtest"]
    st = cfg["port_analysis_config"]["strategy"]["kwargs"]
    summary = {}
    for h in HORIZONS:
        best = max((k for k in results if results[k]["horizon"] == h), key=lambda k: results[k]["valid"]["RankIC"])
        strategy = TopkDropoutStrategy(signal=preds[best], topk=st["topk"], n_drop=st["n_drop"])
        report, _ = backtest_daily(start_time=bt["start_time"], end_time=bt["end_time"], strategy=strategy,
                                   account=bt["account"], benchmark=bt["benchmark"],
                                   exchange_kwargs=bt["exchange_kwargs"])
        report.to_pickle(OUT / f"report_{best}.pkl")
        cum = lambda x: float((1 + x).prod() - 1)
        ex_nc = risk_analysis(report["return"] - report["bench"])["risk"]
        ex_c = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        summary[best] = {
            "cum_return_no_cost": cum(report["return"]), "cum_return_with_cost": cum(report["return"] - report["cost"]),
            "cum_bench": cum(report["bench"]), "avg_daily_turnover": float(report["turnover"].mean()),
            "total_cost": float(report["cost"].sum()),
            "excess_no_cost": {k: float(ex_nc[k]) for k in ["annualized_return", "information_ratio", "max_drawdown"]},
            "excess_with_cost": {k: float(ex_c[k]) for k in ["annualized_return", "information_ratio", "max_drawdown"]},
            "by_year_with_cost": {int(y): round(cum(g["return"] - g["cost"]), 4) for y, g in report.groupby(report.index.year)},
            "bench_by_year": {int(y): round(cum(g["bench"]), 4) for y, g in report.groupby(report.index.year)},
        }
        print(best, json.dumps(summary[best], ensure_ascii=False), flush=True)

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump({"grid": results, "backtest": summary}, f, ensure_ascii=False, indent=2)
    print(f"完成，共 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
