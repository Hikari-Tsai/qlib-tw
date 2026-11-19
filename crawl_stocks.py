"""爬取證交所上市個股日資料與除權息／減資／面額變更事件，原始 JSON 快取到 data/raw/。

可中斷續爬：已存在的快取檔會跳過（無交易日也會留下記錄檔，下次不再請求）。
請求間隔預設 3 秒；遇到封鎖或錯誤時指數退避後重試。

用法：
    uv run python crawl_stocks.py                  # 事件表 + 2005-01-03 起至最新交易日
    uv run python crawl_stocks.py --start 20240101 --end 20240110
    uv run python crawl_stocks.py --skip-events
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import logging
import os
import time
from pathlib import Path

import requests
import truststore

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent
RAW = Path(os.environ.get("QTW_RAW") or ROOT / "data" / "raw")
DAILY_DIR = RAW / "daily"
EVENT_DIR = RAW / "events"
LOG_PATH = ROOT / "data" / "crawl.log"
TAIEX_CSV = ROOT.parent / "data" / "taiex_daily.csv"  # 主專案的加權指數（只讀），用來找週六補班交易日

MI_INDEX_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d}&type=ALLBUT0999&response=json"
EVENT_URLS = {
    # 除權除息計算結果表
    "TWT49U": "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={s}&endDate={e}&response=json",
    # 減資恢復買賣參考價格
    "TWTAUU": "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU?startDate={s}&endDate={e}&response=json",
    # 變更面額恢復買賣參考價格
    "TWTB8U": "https://www.twse.com.tw/rwd/zh/change/TWTB8U?startDate={s}&endDate={e}&response=json",
}
NO_DATA_HINT = "沒有符合條件"

log = logging.getLogger("crawl")


class Fetcher:
    def __init__(self, interval: float):
        self.interval = interval
        self.last = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    def get_json(self, url: str) -> dict:
        backoff = 60
        for attempt in range(1, 9):
            wait = self.interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            try:
                r = self.session.get(url, timeout=60)
                r.raise_for_status()
                j = r.json()
                if "stat" not in j:
                    raise ValueError(f"缺少 stat 欄位: {str(j)[:200]}")
                return j
            except Exception as e:  # 封鎖時常回傳 HTML 或連線被拒
                log.warning("請求失敗 (第 %d 次) %s: %s；%d 秒後重試", attempt, url, e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 1800)
        raise RuntimeError(f"多次重試仍失敗: {url}")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0、不寫檔名：內容相同時檔案位元完全相同，git 不會把沒變的事件表當成新版本
    data = gzip.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), mtime=0)
    if path.exists() and path.read_bytes() == data:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def daily_path(d: dt.date) -> Path:
    return DAILY_DIR / f"{d.year}" / f"{d:%Y%m%d}.json.gz"


def crawl_events(fetcher: Fetcher, start_year: int, end_year: int, refresh_last: bool) -> None:
    for name, tpl in EVENT_URLS.items():
        for y in range(start_year, end_year + 1):
            path = EVENT_DIR / f"{name}_{y}.json.gz"
            if path.exists() and not (refresh_last and y == end_year):
                continue
            j = fetcher.get_json(tpl.format(s=f"{y}0101", e=f"{y}1231"))
            save_json(path, j)
            log.info("事件 %s %d: stat=%s 筆數=%d", name, y, j.get("stat"), len(j.get("data", [])))


def crawl_daily(fetcher: Fetcher, start: dt.date, end: dt.date) -> None:
    days = [start + dt.timedelta(i) for i in range((end - start).days + 1)]
    weekend_trading = set()
    if TAIEX_CSV.exists():
        with open(TAIEX_CSV, encoding="utf-8") as f:
            next(f)
            weekend_trading = {dt.date.fromisoformat(line[:10]) for line in f if line[:10]}
        weekend_trading = {d for d in weekend_trading if d.weekday() >= 5}
    days = [d for d in days if d.weekday() < 5 or d in weekend_trading]  # 平日 + 週六補班日
    todo = [d for d in days if not daily_path(d).exists()]
    log.info("日資料：%s ~ %s 共 %d 天（平日＋補班日），待爬 %d", start, end, len(days), len(todo))
    t0 = time.monotonic()
    n_trade = 0
    for i, d in enumerate(todo, 1):
        j = fetcher.get_json(MI_INDEX_URL.format(d=f"{d:%Y%m%d}"))
        has_data = j.get("stat") == "OK" and any(len(t.get("data", [])) > 100 for t in j.get("tables", []))
        if has_data:
            n_trade += 1
            save_json(daily_path(d), j)
        elif NO_DATA_HINT in str(j.get("stat")) and d < dt.date.today():
            save_json(daily_path(d), j)  # 休市日：留下記錄避免重爬
        else:
            log.warning("%s 無資料且非明確休市（stat=%s），不快取", d, j.get("stat"))
        if i % 50 == 0 or i == len(todo):
            el = time.monotonic() - t0
            eta = el / i * (len(todo) - i)
            log.info("進度 %d/%d（最新 %s，交易日 %d）已用 %.0f 分，預估剩 %.0f 分",
                     i, len(todo), d, n_trade, el / 60, eta / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20050101")
    ap.add_argument("--end", default=None, help="預設：今天（15:00 前為昨天）")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--skip-events", action="store_true")
    ap.add_argument("--skip-daily", action="store_true")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()])
    start = dt.datetime.strptime(args.start, "%Y%m%d").date()
    if args.end:
        end = dt.datetime.strptime(args.end, "%Y%m%d").date()
    else:
        now = dt.datetime.now()
        end = now.date() if now.hour >= 15 else now.date() - dt.timedelta(1)

    fetcher = Fetcher(args.interval)
    if not args.skip_events:
        crawl_events(fetcher, start.year, end.year, refresh_last=True)
    if not args.skip_daily:
        crawl_daily(fetcher, start, end)
    log.info("完成")


if __name__ == "__main__":
    main()
