"""每日增量爬蟲（GitHub Actions 用）：補抓上次之後到今天的 MI_INDEX、三大法人、融資融券，以及當年度事件表。

- 一天的三份資料要都拿到才存：法人與融資融券通常比收盤資料晚公布（融資融券約晚上 9 點），
  沒拿齊就整天跳過，下次執行再補。
- 休市日（含週末）留下記錄檔，避免重複請求。
- 路徑跟 build_dataset.py 一樣可用 QTW_RAW 覆寫；請求間隔 3 秒。

用法：QTW_RAW=state/raw uv run python daily_crawl.py [--max-days 30]
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from crawl_stocks import EVENT_URLS, MI_INDEX_URL, NO_DATA_HINT, Fetcher, save_json

ROOT = Path(__file__).resolve().parent
RAW = Path(os.environ.get("QTW_RAW") or ROOT / "data" / "raw")
T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86?date={d}&selectType=ALLBUT0999&response=json"
MARGIN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d}&selectType=ALL&response=json"
log = logging.getLogger("daily_crawl")


def path(kind: str, d: dt.date) -> Path:
    return RAW / kind / f"{d.year}" / f"{d:%Y%m%d}.json.gz"


def rows(j: dict) -> int:
    n = len(j.get("data") or [])
    return max([n] + [len(t.get("data") or []) for t in j.get("tables") or []])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=30, help="最多往回補幾個日曆天")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    today = dt.datetime.now(ZoneInfo("Asia/Taipei")).date()
    cached = sorted(p.name[:8] for p in (RAW / "daily").glob("*/*.json.gz"))
    if not cached:
        raise SystemExit(f"{RAW / 'daily'} 沒有任何快取，請先用 seed_state.py 建立初始資料")
    last = dt.datetime.strptime(cached[-1], "%Y%m%d").date()
    start = max(last + dt.timedelta(days=1), today - dt.timedelta(days=args.max_days))
    days = [start + dt.timedelta(i) for i in range((today - start).days + 1)]
    log.info("上次資料 %s，檢查 %s ~ %s 共 %d 天", last, start, today, len(days))

    f = Fetcher(args.interval)
    added = []
    for d in days:
        ds = f"{d:%Y%m%d}"
        mi = f.get_json(MI_INDEX_URL.format(d=ds))
        if not (mi.get("stat") == "OK" and rows(mi) > 100):
            if NO_DATA_HINT in str(mi.get("stat")) and d < today:
                save_json(path("daily", d), mi)  # 休市日
                log.info("%s 休市", d)
            else:
                log.info("%s 尚無收盤資料（stat=%s），停止", d, mi.get("stat"))
                break
            continue
        t86 = f.get_json(T86_URL.format(d=ds))
        mg = f.get_json(MARGIN_URL.format(d=ds))
        if rows(t86) <= 100 or rows(mg) <= 100:
            log.info("%s 法人或融資融券尚未公布（T86=%s、融資=%s），這天下次再補", d, t86.get("stat"), mg.get("stat"))
            break
        save_json(path("t86", d), t86)
        save_json(path("margin", d), mg)
        save_json(path("daily", d), mi)  # 最後才存日資料，代表這天三份都齊了
        added.append(d)
        log.info("%s 完成（%d 檔）", d, rows(mi))

    # 事件表：當年度每次重抓，前一年度沒有就補
    for name, tpl in EVENT_URLS.items():
        for y in (today.year - 1, today.year):
            p = RAW / "events" / f"{name}_{y}.json.gz"
            if y == today.year or not p.exists():
                save_json(p, f.get_json(tpl.format(s=f"{y}0101", e=f"{y}1231")))
    log.info("新增交易日：%s", [str(d) for d in added] or "無")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(f"added={len(added)}\n")


if __name__ == "__main__":
    main()
