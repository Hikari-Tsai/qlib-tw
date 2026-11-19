"""訓練正式版模型（用到最新資料），存到 models/<版本>/。

與研究用的 train_horizons.py 不同：沒有保留測試期，所有可用資料都拿來訓練。
1. 驗證期 = 最近約一年（預設 2025-07-01 到該週期標籤可用的最後一天），用來
   - 比較 base／flow 兩種特徵，擇一；
   - 以提早停止決定訓練輪數。
2. 再用「訓練期 + 驗證期」全部資料、相同輪數重新訓練，存成正式模型。
標籤要用到未來 H+1 天的價格，所以每個週期能用的資料只到最後一天往前 H+1 個交易日。
訓練用到的最後一天之後，才是這個版本真正的樣本外。

用法（在 qlib_tw 內，需要本機完整資料）：
    uv run python train_production.py --version v2 [--valid-start 2025-07-01]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HORIZONS = [1, 5, 10, 20, 60]
FEATURE_SETS = {"base": "2008-01-01", "flow": "2012-09-01"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--pool", default="tw100")
    ap.add_argument("--valid-start", default="2025-07-01")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    import qlib
    from qlib.data import D

    from run_experiments import PARAMS
    from run_objectives import evaluate
    from tw_qlib.handlers import FLOW_FEATURES, Alpha158Flow

    out = ROOT / "models" / args.version
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} 已存在，請換一個版本名稱")
    out.mkdir(parents=True)
    qlib.init(provider_uri=str(ROOT / "qlib_data"), region="tw", kernels=3)
    cal = pd.DatetimeIndex(D.calendar(start_time="2008-01-01"))
    last = cal[-1]
    vstart = pd.Timestamp(args.valid_start)
    t0 = time.time()

    exprs = [f"Ref($close, -{h + 1})/Ref($close, -1) - 1" for h in HORIZONS]
    handler = Alpha158Flow(
        instruments=args.pool, start_time="2008-01-01", end_time=last,
        fit_start_time="2008-01-01", fit_end_time=last, label=(exprs, [f"LABEL{h}" for h in HORIZONS]),
        learn_processors=[{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
    )
    learn = handler.fetch(col_set=["feature", "label"], data_key="learn")
    raw = handler.fetch(col_set="label", data_key="raw").reindex(learn.index)
    X, Y = learn["feature"].astype(np.float32), learn["label"]
    del learn, handler
    gc.collect()
    cols_of = {"base": [c for c in X.columns if c not in FLOW_FEATURES], "flow": list(X.columns)}
    dates = X.index.get_level_values("datetime")
    print(f"資料載入 {time.time() - t0:.0f} 秒；{X.shape}；資料到 {last:%Y-%m-%d}", flush=True)

    summary = {}
    for h in HORIZONS:
        label_end = cal[-1 - (h + 1)]  # 標籤可用的最後一天
        purge_end = cal[cal.searchsorted(vstart) - (h + 1) - 1]  # 訓練期結尾，留 H+1 天不和驗證期重疊
        y, y_raw = Y[f"LABEL{h}"], raw[f"LABEL{h}"]
        ok = y_raw.notna().to_numpy() & (dates <= label_end)
        va = (dates >= vstart) & ok
        cand = {}
        for fs, start in FEATURE_SETS.items():
            cols = cols_of[fs]
            tr = (dates >= pd.Timestamp(start)) & (dates <= purge_end) & ok
            dtr = lgb.Dataset(X.loc[tr, cols], y[tr])
            dva = lgb.Dataset(X.loc[va, cols], y[va], reference=dtr)
            params = {"objective": "regression", "metric": "l2", "verbosity": -1, "num_threads": 16, "seed": 42,
                      **PARAMS["shallow"]}
            m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            s = pd.Series(m.predict(X.loc[va, cols], num_iteration=m.best_iteration), index=X.index[va])
            cand[fs] = (m.best_iteration, evaluate(s, y_raw[va]), params)
            print(f"H{h:<3d} {fs:5s} iter={m.best_iteration:4d}  驗證 RankIC={cand[fs][1]['RankIC']:.4f}", flush=True)
        best = max(cand, key=lambda k: cand[k][1]["RankIC"])
        n_iter, vmetrics, params = cand[best]
        cols = cols_of[best]
        full = (dates >= pd.Timestamp(FEATURE_SETS[best])) & ok
        model = lgb.train(params, lgb.Dataset(X.loc[full, cols], y[full]), num_boost_round=max(n_iter, 10))
        name = f"{args.pool}_H{h}"
        model.save_model(str(out / f"{name}.txt"))
        meta = {"version": args.version, "pool": args.pool, "horizon": h, "feature_set": best, "features": cols,
                "train": [FEATURE_SETS[best], f"{label_end:%Y-%m-%d}"],
                "valid": [f"{vstart:%Y-%m-%d}", f"{label_end:%Y-%m-%d}"],
                "label_end": f"{label_end:%Y-%m-%d}", "data_end": f"{last:%Y-%m-%d}",
                "best_iter": int(n_iter), "params": PARAMS["shallow"],
                "label": f"Ref($close, -{h + 1})/Ref($close, -1) - 1（每日 rank 標準化）",
                "eval_name": "驗證期（2025-07 起，訓練前選特徵與輪數用）", "eval_metrics": vmetrics,
                "alternative": {k: c[1] for k, c in cand.items() if k != best},
                "note": "最終模型以訓練期＋驗證期全部資料重新訓練，驗證期指標只供參考（偏樂觀）",
                "created": dt.date.today().isoformat()}
        (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[f"H{h}"] = {"feature_set": best, "best_iter": int(n_iter), "valid_rank_ic": round(vmetrics["RankIC"], 4)}
    print(json.dumps(summary, ensure_ascii=False))
    print(f"完成，共 {time.time() - t0:.0f} 秒；模型在 {out}")


if __name__ == "__main__":
    main()
