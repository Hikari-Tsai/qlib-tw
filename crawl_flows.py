"""爬取證交所三大法人買賣超（T86）與融資融券餘額（MI_MARGN），原始 JSON 快取到 data/raw/{t86,margin}/。

- 只爬股價資料裡有的交易日（qlib_data/calendars/day.txt）。
- T86 從 2012-05-02 才有全市場資料；融資融券從 2005 起。
- 爬取順序：先 2012-05-02 ~ 2021-11-04（訓練／驗證／測試會用到），再到資料結尾，最後補 2005 ~ 2012 的融資融券。
- 與 crawl_stocks.py 共用 Fetcher：請求間隔 3 秒，失敗指數退避；已快取的日期跳過，可中斷續爬。

用法（在 qlib_tw 內）：uv run python crawl_flows.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

from crawl_stocks import NO_DATA_HINT, Fetcher, save_json

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
LOG_PATH = ROOT / "data" / "crawl_flows.log"
CALENDAR = ROOT / "qlib_data" / "calendars" / "day.txt"

SOURCES = {
    "t86": ("https://www.twse.com.tw/rwd/zh/fund/T86?date={d}&selectType=ALLBUT0999&response=json",
            dt.date(2012, 5, 2)),
    "margin": ("https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d}&selectType=ALL&response=json",
               dt.date(2005, 1, 3)),
}
PRIORITY_START, PRIORITY_END = dt.date(2012, 5, 2), dt.date(2021, 11, 4)

log = logging.getLogger("crawl_flows")


def cache_path(name: str, d: dt.date) -> Path:
    return RAW / name / f"{d.year}" / f"{d:%Y%m%d}.json.gz"


def has_rows(j: dict) -> bool:
    if j.get("stat") != "OK":
        return False
    if len(j.get("data", [])) > 100:
        return True
    return any(len(t.get("data") or []) > 100 for t in j.get("tables", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()])

    cal = [dt.date.fromisoformat(x.strip()) for x in CALENDAR.read_text().split() if x.strip()]
    order = (lambda d: (0 if PRIORITY_START <= d <= PRIORITY_END else 1 if d > PRIORITY_END else 2, d))
    todo = sorted(((n, d) for n, (_, first) in SOURCES.items() for d in cal
                   if d >= first and not cache_path(n, d).exists()),
                  key=lambda x: (order(x[1]), x[0]))
    log.info("待爬 %d 筆（%s）", len(todo), {n: sum(1 for m, _ in todo if m == n) for n in SOURCES})

    fetcher = Fetcher(args.interval)
    t0 = time.monotonic()
    for i, (name, d) in enumerate(todo, 1):
        j = fetcher.get_json(SOURCES[name][0].format(d=f"{d:%Y%m%d}"))
        if has_rows(j) or NO_DATA_HINT in str(j.get("stat")):
            save_json(cache_path(name, d), j)  # 沒資料也留記錄，避免重爬
            if not has_rows(j):
                log.warning("%s %s 無資料（stat=%s）", name, d, j.get("stat"))
        else:
            log.warning("%s %s 非預期回應（stat=%s），不快取", name, d, j.get("stat"))
        if i % 100 == 0 or i == len(todo):
            el = time.monotonic() - t0
            log.info("進度 %d/%d（%s %s）已用 %.0f 分，預估剩 %.0f 分",
                     i, len(todo), name, d, el / 60, el / i * (len(todo) - i) / 60)


if __name__ == "__main__":
    main()
