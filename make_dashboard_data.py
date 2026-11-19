"""把各輪實驗結果整理成儀表板用的 JSON（每個檔案 {rows, meta}，meta 帶本腳本原文）。

輸入：outputs/{experiments,exclude_losers,objectives,flows}/ 的 results.json、report_*.pkl、pred_*.pkl
輸出：outputs/dashboard/*.json

用法（在 qlib_tw 內）：uv run python make_dashboard_data.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DST = OUT / "dashboard"
CODE = Path(__file__).read_text(encoding="utf-8")
H = 20


def load(name: str) -> dict:
    return json.loads((OUT / name / "results.json").read_text(encoding="utf-8"))


def clean(v):
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    return v


def write(name: str, rows: list[dict], inputs: str) -> None:
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    meta = [{"label": "make_dashboard_data.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": inputs}]
    (DST / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {len(rows)} 列")


def bt_row(round_, pool, model, strategy, split, m) -> dict:
    return {"round": round_, "pool": pool, "model": model, "strategy": strategy, "split": split,
            "cum_with_cost": m["cum_return_with_cost"], "cum_bench": m["cum_bench"],
            "excess_ann": m["excess_with_cost"]["annualized_return"],
            "ir": m["excess_with_cost"]["information_ratio"], "turnover": m["avg_daily_turnover"]}


def strategy_label(key: str) -> str:
    if key.startswith("exc0_") or key.startswith("ew_"):
        return f"股票池等權（每 {key.split('reb')[1]} 日）"
    if key.startswith("exc"):
        return f"剔除最差 {key[3:key.index('_')]}%（每 {key.split('reb')[1]} 日）"
    return "TopkDropout 前 20 名"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    exp, exc, obj, flo = load("experiments"), load("exclude_losers"), load("objectives"), load("flows")

    # 1. 標籤期間 × 參數
    write("grid", [{"horizon": f"{r['horizon']} 天", "params": r["params"], "best_iter": r["best_iter"],
                    "valid_rank_ic": r["valid"]["RankIC"], "test_rank_ic": r["test"]["RankIC"],
                    "test_rank_icir": r["test"]["RankICIR"]} for r in exp["grid"].values()],
          "outputs/experiments/results.json")

    # 2. 模型訊號：股票池 × 目標、法人特徵
    models = []
    for pool, d in obj.items():
        for o, r in d["models"].items():
            models.append({"round": "訓練目標", "pool": pool, "model": o, "valid_rank_ic": r["valid"]["RankIC"],
                           "test_rank_ic": r["test"]["RankIC"], "test_bottom20": r["test"]["bottom20_spread"],
                           "test_top20": r["test"]["top20_spread"]})
    for pool, d in flo.items():
        for o, r in d["models"].items():
            models.append({"round": "法人籌碼", "pool": pool, "model": o, "valid_rank_ic": r["valid"]["RankIC"],
                           "test_rank_ic": r["test"]["RankIC"], "test_bottom20": r["test"]["bottom20_spread"],
                           "test_top20": r["test"]["top20_spread"]})
    write("models", models, "outputs/objectives/results.json, outputs/flows/results.json")

    # 3. 回測總表（驗證期與測試期，扣成本）
    bts = []
    for key, m in exp["backtest"].items():
        bts.append(bt_row("標籤期間", "tw100", key, "TopkDropout 前 20 名", "test", m))
    for split in ("valid", "test"):
        for key, m in exc[split].items():
            bts.append(bt_row("排除輸家", "tw100", "H20_shallow", strategy_label(key), split, m))
    for pool, d in obj.items():
        for split in ("valid", "test"):
            for key, m in d[f"{split}_backtest"].items():
                bts.append(bt_row("訓練目標", pool, d["chosen_objective"], strategy_label(key), split, m))
    for pool, d in flo.items():
        for split in ("valid", "test"):
            for key, m in d[f"{split}_backtest"].items():
                bts.append(bt_row("法人籌碼", pool, "flow", strategy_label(key), split, m))
    write("backtests", bts, "outputs/*/results.json")

    # 4. 測試期累積報酬曲線（扣成本）與逐年報酬
    reports = {
        "TopkDropout 前 20 名": OUT / "experiments" / "report_H20_shallow.pkl",
        "剔除最差 20%": OUT / "exclude_losers" / "report_test_exc20_reb5.pkl",
        "股票池等權": OUT / "exclude_losers" / "report_test_exc0_reb5.pkl",
    }
    curves, yearly = [], []
    bench = None
    for name, p in reports.items():
        r = pd.read_pickle(p)
        net = r["return"] - r["cost"]
        bench = r["bench"]
        for d, v in (1 + net).cumprod().sub(1).items():
            curves.append({"date": f"{d:%Y-%m-%d}", "series": name, "cum_return": float(v)})
        for y, g in net.groupby(net.index.year):
            yearly.append({"series": name, "year": str(y), "return": float((1 + g).prod() - 1)})
    for d, v in (1 + bench).cumprod().sub(1).items():
        curves.append({"date": f"{d:%Y-%m-%d}", "series": "加權指數", "cum_return": float(v)})
    for y, g in bench.groupby(bench.index.year):
        yearly.append({"series": "加權指數", "year": str(y), "return": float((1 + g).prod() - 1)})
    write("curves", curves, ", ".join(str(p.relative_to(ROOT)) for p in reports.values()))
    write("yearly", yearly, ", ".join(str(p.relative_to(ROOT)) for p in reports.values()))

    # 5. 預測五分位的平均 20 日報酬（測試期，標籤不跨進保留集）
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(ROOT / "qlib_data"), region="tw", kernels=3)
    cal = D.calendar(start_time="2019-01-01", end_time="2021-11-05")
    last = cal[cal.searchsorted(pd.Timestamp("2021-11-05")) - (H + 1) - 1]
    quint = []
    for label, pool, pred_path in [("tw100 價量模型", "tw100", OUT / "experiments" / "pred_H20_shallow.pkl"),
                                   ("tw300 價量＋法人籌碼", "tw300", OUT / "flows" / "pred_tw300_flow.pkl")]:
        pred = pd.read_pickle(pred_path)
        pred = pred[(pred.index.get_level_values("datetime") >= pd.Timestamp("2019-01-01"))
                    & (pred.index.get_level_values("datetime") <= last)]
        lab = D.features(D.instruments(pool), [f"Ref($close, -{H + 1})/Ref($close, -1) - 1"],
                         start_time="2019-01-01", end_time=f"{last:%Y-%m-%d}").iloc[:, 0]
        lab = lab.swaplevel().sort_index()
        df = pd.DataFrame({"p": pred, "y": lab.reindex(pred.index)}).dropna()
        df["q"] = df.groupby(level="datetime")["p"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        by_day = df.groupby([df.index.get_level_values("datetime"), "q"])["y"].mean().unstack()
        pool_mean = df.groupby(level="datetime")["y"].mean().mean()
        names = ["最低", "2", "3", "4", "最高"]
        for q in range(5):
            quint.append({"model": label, "quintile": names[q], "avg_return_20d": float(by_day[q].mean()),
                          "vs_pool": float(by_day[q].mean() - pool_mean)})
    write("quintiles", quint, "outputs/experiments/pred_H20_shallow.pkl, outputs/flows/pred_tw300_flow.pkl, qlib_data")


if __name__ == "__main__":
    main()
