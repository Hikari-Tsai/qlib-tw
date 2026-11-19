"""把 crawl_flows.py 的快取轉成 Qlib 欄位，直接寫進 qlib_data/features/<代號>/<欄位>.day.bin。

欄位（股數都除以 factor，和 Qlib 的 $volume 同一單位，可以直接相除）：
- $foreign_net：外資（含外資自營商）買賣超 = 三大法人 − 投信 − 自營商
- $trust_net：投信買賣超
- $dealer_net：自營商買賣超（自行買賣＋避險）
- $margin_bal：融資餘額（張 × 1000 → 股）
- $short_bal：融券餘額（張 × 1000 → 股）

規則：
- 有快取的日期：當天沒出現在表上的股票記 0（沒有法人交易／沒有信用餘額）。
- 沒有快取的日期（T86 在 2012-05-02 之前、或尚未爬到）：NaN。
- 每檔股票的欄位起訖和它的 $close 相同（同一個 bin 起始索引）。

用法（在 qlib_tw 內）：uv run python build_flows.py
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = Path(os.environ.get("QTW_RAW") or ROOT / "data" / "raw")
QLIB = Path(os.environ.get("QTW_QLIB") or ROOT / "qlib_data")
FIELDS = ["foreign_net", "trust_net", "dealer_net", "margin_bal", "short_bal"]


def num(x: str) -> float:
    try:
        return float(str(x).replace(",", "").strip() or "nan")
    except ValueError:
        return np.nan


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def parse_t86(j: dict) -> dict[str, tuple[float, float, float]]:
    if j.get("stat") != "OK":
        return {}
    f = j["fields"]
    i_total = next(i for i, n in enumerate(f) if n.startswith("三大法人買賣超"))
    i_trust = next(i for i, n in enumerate(f) if n.startswith("投信買賣超"))
    # 自營商買賣超：2014-12 前只有一欄；之後是「自營商買賣超股數」總計＋自行買賣／避險分欄
    i_dealer = next(i for i, n in enumerate(f) if n == "自營商買賣超股數" or n.startswith("自營商買賣超股數") and "(" not in n)
    out = {}
    for r in j["data"]:
        code = r[0].strip()
        if re.fullmatch(r"[1-9]\d{3}", code):
            tot, tru, dea = num(r[i_total]), num(r[i_trust]), num(r[i_dealer])
            out[code] = (tot - tru - dea, tru, dea)
    return out


def parse_margin(j: dict) -> dict[str, tuple[float, float]]:
    if j.get("stat") != "OK":
        return {}
    for t in j.get("tables", []):
        f = t.get("fields") or []
        if len(t.get("data") or []) > 100 and f[:2] == ["代號", "名稱"]:
            # 欄位：代號 名稱 | 融資 買進 賣出 現金償還 前日餘額 今日餘額 限額 | 融券 買進 賣出 現券償還 前日餘額 今日餘額 限額 | ...
            return {r[0].strip(): (num(r[6]) * 1000, num(r[12]) * 1000)
                    for r in t["data"] if re.fullmatch(r"[1-9]\d{3}", r[0].strip())}
    return {}


def build_frame(name: str, parser, cols: list[str], calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    files = sorted((RAW / name).glob("*/*.json.gz"))
    recs, covered = [], []
    for p in files:
        d = pd.Timestamp(p.name[:8])
        rows = parser(load(p))
        if not rows:
            continue
        covered.append(d)
        recs += [(d, c, *v) for c, v in rows.items()]
    df = pd.DataFrame(recs, columns=["date", "code", *cols])
    covered = pd.DatetimeIndex(sorted(covered))
    out = {}
    for c in cols:
        w = df.pivot(index="date", columns="code", values=c).reindex(calendar)
        w.loc[covered] = w.loc[covered].fillna(0.0)  # 有快取的日子：沒出現代表 0
        out[c] = w
    print(f"{name}: {len(files)} 個快取檔、{len(covered)} 天有資料（{covered.min():%Y-%m-%d} ~ {covered.max():%Y-%m-%d}）")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    calendar = pd.DatetimeIndex(pd.read_csv(QLIB / "calendars" / "day.txt", header=None)[0])
    wide = build_frame("t86", parse_t86, ["foreign_net", "trust_net", "dealer_net"], calendar)
    wide |= build_frame("margin", parse_margin, ["margin_bal", "short_bal"], calendar)
    n = 0
    for d in sorted((QLIB / "features").iterdir()):
        code = d.name
        if not (d / "factor.day.bin").exists():  # 指數（TWII）沒有這些欄位
            continue
        close = np.fromfile(d / "close.day.bin", dtype="<f4")
        start, length = int(close[0]), len(close) - 1
        fac = np.fromfile(d / "factor.day.bin", dtype="<f4")[1:]
        for f in FIELDS:
            w = wide[f]
            vals = (w[code].to_numpy(dtype=np.float64)[start:start + length] if code in w.columns
                    else np.where(w.notna().any(axis=1).to_numpy()[start:start + length], 0.0, np.nan))
            vals = vals / fac  # 和 $volume 一樣以還原後股數表示
            np.concatenate([[start], vals]).astype("<f4").tofile(d / f"{f}.day.bin")
        n += 1
    print(f"寫入 {n} 檔股票 × {len(FIELDS)} 個欄位")


if __name__ == "__main__":
    main()
