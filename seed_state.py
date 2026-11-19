"""建立 GitHub Actions 用的初始狀態 state/：把本機完整資料的最近 N 個交易日原始檔複製到 state/raw/。

state/history/ 由 history_update.py（用本機完整資料回填 2019 年起）產生；本腳本只處理 state/raw/。

用法（在 qlib_tw 內）：uv run python seed_state.py [--keep 200]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "raw"
DST = ROOT / "state" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=200, help="保留的交易日數")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    cal = [x.strip() for x in (ROOT / "qlib_data" / "calendars" / "day.txt").read_text().split() if x.strip()]
    start = cal[-args.keep].replace("-", "")
    if DST.exists():
        shutil.rmtree(DST)
    n = 0
    for kind in ("daily", "t86", "margin"):
        for p in sorted((SRC / kind).glob("*/*.json.gz")):
            if p.name[:8] >= start:  # 含期間內的休市記錄檔
                q = DST / kind / p.parent.name / p.name
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
                n += 1
    years = {int(start[:4]), int(cal[-1][:4])}
    for p in (SRC / "events").glob("*.json.gz"):
        if int(p.stem.split("_")[1].split(".")[0]) >= min(years):
            q = DST / "events" / p.name
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            n += 1
    size = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"state/raw：{start} ~ {cal[-1].replace('-', '')}，{n} 個檔案，{size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
