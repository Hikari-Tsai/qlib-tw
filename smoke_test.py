"""冒煙測試：少量股票、短期間，確認 Qlib 能讀資料、Alpha158 算得出來、設定檔可載入。不訓練模型。

用法：uv run python smoke_test.py [--start 2019-01-01 --end 2020-12-31]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parent
CODES = ["2330", "2412", "2317", "1301", "2882"]


def check(cond: bool, msg: str) -> None:
    print(("  [OK] " if cond else "  [FAIL] ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2020-12-31")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = YAML(typ="safe", pure=True).load((ROOT / "workflow_tw_alpha158_lgbm.yaml").read_text(encoding="utf-8"))
    print("1. 設定檔可解析：market =", cfg["market"], "benchmark =", cfg["benchmark"])
    qlib.init(**{**cfg["qlib_init"], "provider_uri": str(ROOT / "qlib_data")})

    print("2. 日曆與股票清單")
    cal = D.calendar(start_time=args.start, end_time=args.end)
    check(len(cal) > 100, f"交易日 {len(cal)} 天（{cal[0]:%Y-%m-%d} ~ {cal[-1]:%Y-%m-%d}）")
    full_cal = D.calendar()
    print(f"  全期間日曆 {full_cal[0]:%Y-%m-%d} ~ {full_cal[-1]:%Y-%m-%d}，{len(full_cal)} 天")
    all_codes = D.list_instruments(D.instruments("all"), start_time=args.start, end_time=args.end, as_list=True)
    check(len(all_codes) > 500 and "TWII" not in all_codes, f"all 期間內 {len(all_codes)} 檔，不含 TWII")
    pool = D.list_instruments(D.instruments(cfg["market"]), start_time=args.start, end_time=args.end, as_list=True)
    check(100 <= len(pool) <= 400, f"{cfg['market']} 期間內出現過 {len(pool)} 檔")

    print("3. D.features")
    fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap", "$factor", "$change", "$limit_up", "$limit_down"]
    df = D.features(CODES, fields, start_time=args.start, end_time=args.end)
    print(df.groupby(level=0).size().to_string())
    check(set(df.index.get_level_values(0)) == set(CODES), "5 檔皆有資料")
    check(df["$close"].notna().mean() > 0.98, f"收盤價非缺值比例 {df['$close'].notna().mean():.3f}")
    raw_close = (df["$close"] / df["$factor"]).xs("2330")
    print(f"  2330 原始收盤價（close/factor）樣本：\n{raw_close.iloc[[0, -1]].round(2).to_string()}")
    amt_ratio = (df["$close"] * df["$volume"] / df["$amount"]).dropna()
    check(amt_ratio.between(0.85, 1.15).mean() > 0.97, f"還原價×還原量 ≈ 成交金額（中位數 {amt_ratio.median():.3f}）")
    bench = D.features([cfg["benchmark"]], ["$close"], start_time=args.start, end_time=args.end)
    check(len(bench) == len(cal), f"benchmark {cfg['benchmark']} 有 {len(bench)} 筆")

    print("4. Alpha158 特徵（小範圍）")
    from qlib.contrib.data.handler import Alpha158
    h = Alpha158(instruments=CODES, start_time=args.start, end_time=args.end,
                 fit_start_time=args.start, fit_end_time=args.end)
    feat = h.fetch(col_set="feature")
    label = h.fetch(col_set="label")
    check(feat.shape[1] == 158, f"特徵 {feat.shape}，標籤 {label.shape}")
    nan_share = feat.isna().mean().mean()
    check(nan_share < 0.1, f"特徵缺值比例 {nan_share:.3%}")
    print(feat.iloc[-3:, :6].round(4).to_string())

    print("5. 台股交易所（漲跌停旗標、交易單位）")
    ex_cfg = cfg["port_analysis_config"]["backtest"]["exchange_kwargs"]["exchange"]
    kw = dict(ex_cfg["kwargs"], start_time=args.start, end_time=args.end)  # codes 用設定檔的 tw100
    from qlib.utils import init_instance_by_config
    ex = init_instance_by_config({**ex_cfg, "kwargs": kw})
    q = ex.quote_df
    check(ex.trade_unit == 1000 and ex.limit_type == ex.LT_TP_EXP, f"trade_unit={ex.trade_unit}, limit_type={ex.limit_type}")
    print(f"  期間內漲停禁買 {int(q['limit_buy'].sum())} 筆、跌停禁賣 {int(q['limit_sell'].sum())} 筆（共 {len(q)} 筆）")
    print("\n冒煙測試通過")


if __name__ == "__main__":
    main()
