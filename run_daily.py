"""每日更新流程（GitHub Actions 與本機皆可執行）。

1. daily_crawl.py：補抓新交易日的行情、三大法人、融資融券、事件表 → state/raw/
2. build_dataset.py + build_flows.py：只用 state/raw/ 的近期資料建 Qlib 資料 → work/（暫存，不保存）
3. history_update.py：用固定模型替新交易日排名，附加到 state/history/（只新增、不改寫）
4. 刪掉 state/raw/ 中超過保留天數的舊檔
5. build_site.py：從 state/history/ 產生 site/data.js 與 site/api/

用法（在 qlib_tw 內）：uv run python run_daily.py [--skip-crawl] [--keep 200]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
WORK = ROOT / "work"


def run(script: str, env: dict, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args], check=True, env=env, cwd=ROOT)


def prune(raw: Path, keep: int) -> None:
    days = sorted(p for p in (raw / "daily").glob("*/*.json.gz"))
    trading = [p for p in days if p.stat().st_size > 2000]  # 休市記錄檔很小
    if len(trading) <= keep:
        return
    cutoff = trading[-keep].name[:8]
    removed = 0
    for kind in ("daily", "t86", "margin"):
        for p in (raw / kind).glob("*/*.json.gz"):
            if p.name[:8] < cutoff:
                p.unlink()
                removed += 1
    for d in raw.glob("*/*"):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print(f"刪除 {cutoff} 之前的 {removed} 個舊檔")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-crawl", action="store_true")
    ap.add_argument("--keep", type=int, default=200, help="state/raw 保留的交易日數")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    env = os.environ | {
        "PYTHONUTF8": "1", "OPENBLAS_NUM_THREADS": "1",
        "QTW_RAW": str(STATE / "raw"), "QTW_HISTORY": str(STATE / "history"),
        "QTW_QLIB": str(WORK / "qlib_data"), "QTW_PROCESSED": str(WORK / "processed"),
        "QTW_CSV": str(WORK / "csv"), "QTW_REPORT": str(WORK / "quality_report.json"),
        "QTW_TAIEX_CSV": str(WORK / "no_taiex.csv"),  # 不讀主專案的指數檔，改從日資料快取取加權指數
    }
    if not args.skip_crawl:
        run("daily_crawl.py", env)
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "processed").mkdir(parents=True)
    run("build_dataset.py", env)
    run("build_flows.py", env)
    run("history_update.py", env)
    prune(STATE / "raw", args.keep)
    run("build_site.py", env)


if __name__ == "__main__":
    main()
