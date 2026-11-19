"""用 train_horizons.py 存下的模型，對某個交易日的股票池做各週期排序（研究用分數，不是買賣建議）。

用法（在 qlib_tw 內）：
    uv run python rank_stocks.py                     # 資料最後一個交易日、tw100
    uv run python rank_stocks.py --date 2026-11-05 --pool tw100
輸出：outputs/rankings/<pool>_<date>.csv
欄位：code, name, 每個週期的 score_H<H>（當天相對分數，數值本身無單位）與 rank_H<H>（1 = 當天分數最高）

注意：分數只代表同一天股票池內的相對順序。模型只在 2019-01 ~ 2021-11 的測試期評估過，
2021-11-05 之後的表現尚未驗證；測試期依分數選股扣成本後並未贏過股票池等權持有。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"
OUT = ROOT / "outputs" / "rankings"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tw100")
    ap.add_argument("--date", default=None, help="交易日 YYYY-MM-DD，預設為資料最後一天")
    ap.add_argument("--version", default=None, help="模型版本，預設為 models/versions.json 的 default")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    version = args.version or json.loads((MODELS_ROOT / "versions.json").read_text(encoding="utf-8"))["default"]
    MODELS = MODELS_ROOT / version

    import qlib
    from qlib.data import D

    from tw_qlib.handlers import Alpha158Flow

    qlib.init(provider_uri=str(ROOT / "qlib_data"), region="tw", kernels=3)
    cal = D.calendar()
    day = pd.Timestamp(args.date) if args.date else cal[-1]
    if day not in set(cal):
        raise SystemExit(f"{day:%Y-%m-%d} 不是資料中的交易日（資料到 {cal[-1]:%Y-%m-%d}）")
    metas = sorted((json.loads(p.read_text(encoding="utf-8")) for p in MODELS.glob(f"{args.pool}_H*.json")),
                   key=lambda m: m["horizon"])
    if not metas:
        raise SystemExit(f"找不到 models/{args.pool}_H*.json，請先執行 train_horizons.py --pool {args.pool}")

    # 特徵只需要當天；Qlib 會自動往前多讀計算 60 日指標所需的資料
    handler = Alpha158Flow(instruments=args.pool, start_time=day, end_time=day,
                           fit_start_time=day, fit_end_time=day, learn_processors=[])
    X = handler.fetch(col_set="feature").astype(np.float32).xs(day, level="datetime")
    out = pd.DataFrame(index=X.index)
    for m in metas:
        h = m["horizon"]
        model = lgb.Booster(model_file=str(MODELS / f"{args.pool}_H{h}.txt"))
        s = pd.Series(model.predict(X[m["features"]]), index=X.index)
        out[f"score_H{h}"] = s.round(5)
        out[f"rank_H{h}"] = s.rank(ascending=False, method="min").astype(int)

    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel.parquet", columns=["date", "code", "name"])
    names = panel[panel["date"] <= day].drop_duplicates("code", keep="last").set_index("code")["name"].str.strip()
    out.insert(0, "name", names.reindex(out.index))
    out.index.name = "code"
    out = out.sort_values(f"rank_H{20 if any(m['horizon'] == 20 for m in metas) else metas[0]['horizon']}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.pool}_{version}_{day:%Y-%m-%d}.csv"
    out.to_csv(path, encoding="utf-8-sig")
    print(f"{day:%Y-%m-%d} {args.pool}：{len(out)} 檔，週期 {[m['horizon'] for m in metas]}，"
          f"特徵 {[m['feature_set'] for m in metas]}；缺值 {int(X.isna().all(axis=1).sum())} 檔\n輸出 {path}")


if __name__ == "__main__":
    main()
