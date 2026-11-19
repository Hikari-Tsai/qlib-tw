"""訓練不同預測週期的排序模型（每個週期一個獨立的 LightGBM），存檔供 rank_stocks.py 使用。

- 週期 H ∈ {1, 5, 10, 20, 60} 個交易日；標籤是「t+1 收盤買進、持有 H 天」的報酬，每日橫截面 rank 標準化。
- 每個週期比較兩種特徵，依驗證期 Rank IC 擇一：
  - base：Alpha158，2008-01 起訓練；
  - flow：Alpha158 + 三大法人／融資融券 18 個特徵，2012-09 起訓練（法人資料 2012-05 才有）。
- 切分：訓練到 2016 年底、驗證 2017–2018、測試 2019-01 ~ 2021-11-04，各段交界往前留 H+1 天，標籤不跨段。
  2021-11-05 起的保留集不評估（標籤不讀取、不計算任何指標）。
- 參數：run_experiments.py 的 shallow（先前每個週期驗證期都選到它）。

用法（在 qlib_tw 內）：uv run python train_horizons.py [--pool tw100]
輸出：
- models/<pool>_H<H>.txt（LightGBM 模型）與 models/<pool>_H<H>.json（特徵、期間、指標）
- outputs/horizons/<pool>_results.json、pred_<pool>_H<H>.pkl（驗證＋測試期分數）
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "v1"  # 研究版模型；正式版用 train_production.py
OUT = ROOT / "outputs" / "horizons"
HORIZONS = [1, 5, 10, 20, 60]
FEATURE_SETS = {"base": "2008-01-01", "flow": "2012-09-01"}  # 特徵組 → 訓練起始日
VALID_START, TEST_START, HOLDOUT_START, TEST_END = "2017-01-01", "2019-01-01", "2021-11-05", "2021-11-04"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tw100")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    import qlib
    from qlib.contrib.data.handler import Alpha158
    from qlib.data import D
    from ruamel.yaml import YAML

    from run_experiments import PARAMS
    from run_objectives import evaluate
    from tw_qlib.handlers import FLOW_FEATURES, Alpha158Flow

    MODELS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = YAML(typ="safe", pure=True).load((ROOT / "workflow_tw_alpha158_lgbm.yaml").read_text(encoding="utf-8"))
    qlib.init(**{**cfg["qlib_init"], "provider_uri": str(ROOT / "qlib_data"), "kernels": 3})
    cal = D.calendar(start_time="2008-01-01", end_time=HOLDOUT_START)
    seg_end = lambda b, h: cal[cal.searchsorted(pd.Timestamp(b)) - (h + 1) - 1]
    t0 = time.time()

    # 一次讀入所有週期的標籤；Alpha158Flow 的特徵包含 Alpha158 全部欄位，base 只取前 158 欄
    exprs = [f"Ref($close, -{h + 1})/Ref($close, -1) - 1" for h in HORIZONS]
    names = [f"LABEL{h}" for h in HORIZONS]
    handler = Alpha158Flow(
        instruments=args.pool, start_time="2008-01-01", end_time=TEST_END,
        fit_start_time="2008-01-01", fit_end_time="2016-12-31", label=(exprs, names),
        learn_processors=[{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
    )
    learn = handler.fetch(col_set=["feature", "label"], data_key="learn")
    raw = handler.fetch(col_set="label", data_key="raw").reindex(learn.index)
    X, Y = learn["feature"].astype(np.float32), learn["label"]
    del learn, handler
    gc.collect()
    base_cols = [c for c in X.columns if c not in FLOW_FEATURES]
    cols_of = {"base": base_cols, "flow": list(X.columns)}
    dates = X.index.get_level_values("datetime")
    print(f"資料載入 {time.time() - t0:.0f} 秒；{X.shape}", flush=True)

    results = {}
    for h in HORIZONS:
        y, y_raw = Y[f"LABEL{h}"], raw[f"LABEL{h}"]
        ok = y_raw.notna().to_numpy()
        va = (dates >= pd.Timestamp(VALID_START)) & (dates <= seg_end(TEST_START, h)) & ok
        use = dates >= pd.Timestamp(VALID_START)
        te = (dates >= pd.Timestamp(TEST_START)) & (dates <= seg_end(HOLDOUT_START, h))
        cand = {}
        for fs, start in FEATURE_SETS.items():
            tr = (dates >= pd.Timestamp(start)) & (dates <= seg_end(VALID_START, h)) & ok
            cols = cols_of[fs]
            dtr = lgb.Dataset(X.loc[tr, cols], y[tr])
            dva = lgb.Dataset(X.loc[va, cols], y[va], reference=dtr)
            model = lgb.train({"objective": "regression", "metric": "l2", "verbosity": -1, "num_threads": 16,
                               "seed": 42, **PARAMS["shallow"]}, dtr, num_boost_round=2000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            s = pd.Series(model.predict(X.loc[use, cols], num_iteration=model.best_iteration), index=X.index[use])
            v, t = evaluate(s[va[use]], y_raw[va]), evaluate(s[te[use]], y_raw[te])
            cand[fs] = (model, s, v, t)
            print(f"H{h:<3d} {fs:5s} iter={model.best_iteration:4d}  valid RankIC={v['RankIC']:.4f}  "
                  f"test RankIC={t['RankIC']:.4f} ICIR={t['RankICIR']:.2f} "
                  f"bot={t['bottom20_spread']:+.4f} top={t['top20_spread']:+.4f}", flush=True)
        best = max(cand, key=lambda k: cand[k][2]["RankIC"])
        model, s, v, t = cand[best]
        name = f"{args.pool}_H{h}"
        model.save_model(str(MODELS / f"{name}.txt"), num_iteration=model.best_iteration)
        s.to_pickle(OUT / f"pred_{name}.pkl")
        meta = {"pool": args.pool, "horizon": h, "feature_set": best, "features": cols_of[best],
                "train": [FEATURE_SETS[best], f"{seg_end(VALID_START, h):%Y-%m-%d}"],
                "valid": [VALID_START, f"{seg_end(TEST_START, h):%Y-%m-%d}"],
                "test": [TEST_START, f"{seg_end(HOLDOUT_START, h):%Y-%m-%d}"],
                "best_iter": int(model.best_iteration), "params": PARAMS["shallow"],
                "label": f"Ref($close, -{h + 1})/Ref($close, -1) - 1（每日 rank 標準化）",
                "valid_metrics": v, "test_metrics": t,
                "alternative": {k: {"valid": c[2], "test": c[3]} for k, c in cand.items() if k != best}}
        (MODELS / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        results[f"H{h}"] = {k: {"best_iter": int(c[0].best_iteration), "valid": c[2], "test": c[3]} for k, c in cand.items()}
        results[f"H{h}"]["chosen"] = best

    # 不同週期的排序有多相似：測試期每日分數的 Spearman 相關，取平均
    preds = {h: pd.read_pickle(OUT / f"pred_{args.pool}_H{h}.pkl") for h in HORIZONS}
    test_idx = preds[HORIZONS[0]].index[preds[HORIZONS[0]].index.get_level_values("datetime") >= pd.Timestamp(TEST_START)]
    df = pd.DataFrame({f"H{h}": p.reindex(test_idx) for h, p in preds.items()})
    corr = df.groupby(level="datetime").corr(method="spearman").groupby(level=1).mean()
    results["rank_corr_between_horizons"] = corr.round(3).to_dict()
    print("週期之間的排序相關（測試期每日平均）：\n" + corr.round(2).to_string(), flush=True)

    (OUT / f"{args.pool}_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，共 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
