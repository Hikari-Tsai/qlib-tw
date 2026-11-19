"""「排除輸家」只做多策略：每次調倉把股票池依 20 日預測排序，剔除最差的一部分，其餘等權持有。

動機：20 日標籤的預測五分位報酬呈倒 U 型（最低組最差、最高組也落後），
模型的優勢在認出輸家，TopkDropout 只買前 20 名用不到這個優勢。

流程：
1. 用 run_experiments.py 中 H20 + shallow 的同一組設定重新訓練（seed 42），存下驗證期與測試期預測。
2. 驗證期（2017–2018）比較：剔除比例 {0, 20%, 40%, 60%} × 調倉間隔 {5, 20} 個交易日。
   剔除比例 0 就是「股票池等權」，不用模型，當作對照組。
3. 只依驗證期結果挑一組（剔除比例 > 0、扣成本後超額報酬資訊比率最高），
   在測試期（2019-01 ~ 2021-11-04）回測，並和同調倉間隔的股票池等權、加權指數比較。
   2021-11-05 起的保留集不評估。

注意：驗證期也是 LightGBM 提早停止用的資料，驗證期預測略為樂觀；選擇時只用來比較策略形式。

用法（在 qlib_tw 內）：uv run python run_exclude_losers.py
結果：outputs/exclude_losers/results.json、report_*.pkl
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import qlib
from qlib.backtest.decision import TradeDecisionWO
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D
from ruamel.yaml import YAML

from run_experiments import PARAMS, VALID_START, TEST_START, HOLDOUT_START

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "exclude_losers"
H = 20
EXCLUDE = [0.0, 0.2, 0.4, 0.6]
REBALANCE = [5, 20]
VALID_END, TEST_END = "2018-12-31", "2021-11-04"


class ExcludeLosersStrategy(WeightStrategyBase):
    """每 rebalance 個交易日調倉一次：剔除預測最低的 exclude 比例，其餘等權。"""

    def __init__(self, *, exclude: float, rebalance: int, **kwargs):
        super().__init__(**kwargs)
        self.exclude, self.rebalance = exclude, rebalance

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        s = score.dropna() if isinstance(score, pd.Series) else score.iloc[:, 0].dropna()
        keep = s[s.rank(pct=True, method="first") > self.exclude].index
        return {code: 1.0 / len(keep) for code in keep} if len(keep) else None

    def generate_trade_decision(self, execute_result=None):
        if self.trade_calendar.get_trade_step() % self.rebalance != 0:
            return TradeDecisionWO([], self)
        return super().generate_trade_decision(execute_result)


def run_backtest(pred: pd.Series, exclude: float, rebalance: int, start: str, end: str, bt: dict) -> pd.DataFrame:
    ex = copy.deepcopy(bt["exchange_kwargs"])
    ex["exchange"]["kwargs"].update(start_time=start, end_time=end)
    strategy = ExcludeLosersStrategy(signal=pred, exclude=exclude, rebalance=rebalance, risk_degree=0.95)
    report, _ = backtest_daily(start_time=start, end_time=end, strategy=strategy, account=bt["account"],
                               benchmark=bt["benchmark"], exchange_kwargs=ex)
    return report


def metrics(report: pd.DataFrame) -> dict:
    cum = lambda x: float((1 + x).prod() - 1)
    ex_c = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
    return {
        "cum_return_with_cost": cum(report["return"] - report["cost"]),
        "cum_bench": cum(report["bench"]),
        "avg_daily_turnover": float(report["turnover"].mean()),
        "total_cost": float(report["cost"].sum()),
        "excess_with_cost": {k: float(ex_c[k]) for k in ["annualized_return", "information_ratio", "max_drawdown"]},
        "by_year_with_cost": {int(y): round(cum(g["return"] - g["cost"]), 4) for y, g in report.groupby(report.index.year)},
        "bench_by_year": {int(y): round(cum(g["bench"]), 4) for y, g in report.groupby(report.index.year)},
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = YAML(typ="safe", pure=True).load((ROOT / "workflow_tw_alpha158_lgbm.yaml").read_text(encoding="utf-8"))
    qlib.init(**{**cfg["qlib_init"], "provider_uri": str(ROOT / "qlib_data"), "kernels": 3})
    hc = cfg["data_handler_config"]
    bt = cfg["port_analysis_config"]["backtest"]
    t0 = time.time()

    pred_file = OUT / "pred_H20_shallow_valid_test.pkl"
    if pred_file.exists():
        pred = pd.read_pickle(pred_file)
    else:
        handler = Alpha158(
            instruments=hc["instruments"], start_time=hc["start_time"], end_time=hc["end_time"],
            fit_start_time=hc["fit_start_time"], fit_end_time=hc["fit_end_time"],
            label=([f"Ref($close, -{H + 1})/Ref($close, -1) - 1"], ["LABEL0"]),
            learn_processors=[{"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
        )
        learn = handler.fetch(col_set=["feature", "label"], data_key="learn")
        X, y = learn["feature"].astype(np.float32), learn["label"]["LABEL0"]
        print(f"資料載入 {time.time() - t0:.0f} 秒；特徵 {X.shape}", flush=True)
        cal = D.calendar(start_time=hc["start_time"], end_time=HOLDOUT_START)
        seg_end = lambda b: cal[cal.searchsorted(pd.Timestamp(b)) - (H + 1) - 1]
        dates = X.index.get_level_values("datetime")
        tr = (dates <= seg_end(VALID_START)) & y.notna().to_numpy()
        va = (dates >= pd.Timestamp(VALID_START)) & (dates <= seg_end(TEST_START)) & y.notna().to_numpy()
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        model = lgb.train({"objective": "regression", "metric": "l2", "verbosity": -1, "num_threads": 16,
                           "seed": 42, **PARAMS["shallow"]}, dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        use = dates >= pd.Timestamp(VALID_START)  # 驗證期＋測試期全部日期（只用到過去特徵）
        pred = pd.Series(model.predict(X[use], num_iteration=model.best_iteration), index=X.index[use])
        pred.to_pickle(pred_file)
        old = pd.read_pickle(ROOT / "outputs" / "experiments" / "pred_H20_shallow.pkl")
        diff = (pred.reindex(old.index) - old).abs().max()
        print(f"best_iter={model.best_iteration}；與先前測試期預測最大差 {diff:.2e}", flush=True)

    results = {"valid": {}, "test": {}}
    for reb in REBALANCE:
        for exc in EXCLUDE:
            key = f"exc{int(exc * 100)}_reb{reb}"
            m = metrics(run_backtest(pred, exc, reb, VALID_START, VALID_END, bt))
            results["valid"][key] = m
            print(f"valid {key:14s} 累積 {m['cum_return_with_cost']:+.3f}  指數 {m['cum_bench']:+.3f}  "
                  f"超額年化 {m['excess_with_cost']['annualized_return']:+.3f}  "
                  f"IR {m['excess_with_cost']['information_ratio']:+.2f}  換手 {m['avg_daily_turnover']:.3f}", flush=True)

    best = max((k for k in results["valid"] if not k.startswith("exc0_")),
               key=lambda k: results["valid"][k]["excess_with_cost"]["information_ratio"])
    reb = int(best.split("reb")[1])
    results["chosen"] = best
    for key, exc in [(best, int(best[3:best.index("_")]) / 100), (f"exc0_reb{reb}", 0.0)]:
        report = run_backtest(pred, exc, reb, TEST_START, TEST_END, bt)
        report.to_pickle(OUT / f"report_test_{key}.pkl")
        m = metrics(report)
        results["test"][key] = m
        print(f"test  {key:14s}", json.dumps(m, ensure_ascii=False), flush=True)

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成，共 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
