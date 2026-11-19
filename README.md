# 台股多週期排名

用微軟開源量化平台 [Qlib](https://github.com/microsoft/qlib) 與 LightGBM，替臺灣證券交易所上市股票做 1／5／10／20／60 個交易日的相對強弱排名。GitHub Actions 每個交易日自動更新排名，並持續統計依排名買賣的歷史成績，發佈成靜態網頁與 JSON API。

> **免責聲明**：本專案僅供科學實驗與研究用途，不構成任何投資建議或買賣推薦。所有排名與績效皆為模型輸出與歷史回測，不保證未來結果；依本專案資訊進行任何交易所產生的盈虧，作者概不負責。

## 網頁內容

- **今日排名**：股票池（每月成交金額前 100 名的上市普通股）每檔的各週期名次、分數、當日開盤價與收盤價。
  - 可以切換排序週期。
  - 可以和上個週期或前一交易日比較名次漲跌。
- **個股面板**：點任一檔股票，顯示：
  - 各週期名次走勢。
  - 收盤價走勢。
  - 近 10 日的三大法人買賣超與融資融券。
- **模型版本切換**：v1 研究版／v2 正式版。v2 在 2026-11-06 之前的成績會標示「看過答案、偏樂觀」。
- **策略成績**（不計手續費）：
  - 三種策略：做多 TopN、多空（做多 TopN、放空 LowN）、做空 LowN。
  - N = 5／10／20。
  - 期間可選近 1 月到 2019 年至今。
  - 對照組：股票池等權持有、加權指數（不含息）；參考：0050（含息，已處理 2025 年分割）。
- **資料 API**：`api/index.json`、`api/latest.json`、`api/stocks/{代號}.json`，每天更新、唯讀。

## 方法摘要

| 項目 | 內容 |
|---|---|
| 資料 | 證交所公開資料：每日行情（2005 起）、三大法人（2012-05 起）、融資融券。股價依除權息、減資、面額變更還原，含已下市股票 |
| 特徵 | Qlib Alpha158（158 個價量因子）；部分週期另加 18 個法人與融資融券特徵 |
| 目標 | 「明天收盤買進、持有 H 個交易日」的報酬在當天股票池中的排名 |
| 模型 | 每個週期一個 LightGBM，兩個版本並行（網頁可切換）：<br>**v1 研究版**：訓練 2008–2016、驗證 2017–2018、測試 2019-01 ～ 2021-11；2021-11-05 起完全未用於訓練或挑選（長期樣本外紀錄）<br>**v2 正式版**：用到 2026-11-05 的全部資料訓練（2025-07 起的一年先用來選特徵與輪數）；2026-11-06 起才是樣本外 |
| 更新 | 模型固定、不每日重訓；每天替每個版本的新交易日排名，排名只新增不改寫 |

測試期的排序能力很弱：20 日模型的 Rank IC 約 0.05，其他週期約 0.02。依排名選股，扣成本後不一定勝過股票池等權持有。各輪實驗的完整數字與結論見 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 每日更新流程

```
GitHub Actions（交易日 21:30、23:30，台灣時間）
  └ run_daily.py
     1. daily_crawl.py     補抓新交易日：行情、三大法人、融資融券、事件表 → state/raw/
     2. build_dataset.py   只用近 200 個交易日的原始資料建 Qlib 資料（還原、股票池）→ work/
        build_flows.py     法人與融資融券欄位 → work/
     3. history_update.py  固定模型替新交易日排名 → 附加到 state/history/
     4. 刪除超過 200 個交易日的舊原始檔
     5. build_site.py      從 state/history/ 計算策略成績，產生 site/data.js 與 site/api/
  └ commit state/ 回 repo，site/ 部署到 GitHub Pages
```

- **只新增、不改寫**：排名寫進 `state/history/` 之後就不會重算。以後換新版模型，過去的成績也不會被改寫。
- **近期資料就夠**：Alpha158 和法人特徵最多回看 60 個交易日，股票池回看 60 日成交金額，所以只保留近 200 個交易日的原始資料。用這份近期資料重建出的分數，與用 2005 年起完整資料算出的一致。
- 一天的行情、法人、融資融券三份資料要都拿到才處理；還沒公布就等下一次執行。

## 模型版本與重新訓練

- 模型放在 `models/<版本>/`（每個週期一個 LightGBM 文字檔與說明 JSON），`models/versions.json` 列出要記錄的版本、樣本外起點與網頁預設版本。
- GitHub Actions 不訓練模型，只用 repo 裡固定的模型排名。
- 建議每年在本機用完整資料重訓一次：

  ```bash
  uv run python train_production.py --version v3   # 訓練到最新資料 → models/v3/
  ```

  然後在 `versions.json` 加上 v3（`oosStart` 設為訓練資料最後一天的下一個交易日），執行 `history_update.py` 回填 v3 的排名，再推上 repo。舊版本會繼續並行記錄，過去的排名不會被改寫。

## 部署到 GitHub

1. 建立新的 repo，把本資料夾的內容推上去。`.gitignore` 已排除本機的完整資料與實驗輸出。
2. 到 repo 的 **Settings → Pages**，Source 選 **GitHub Actions**。
3. 到 **Actions** 分頁啟用 workflow。可以按 **Run workflow** 手動跑一次，確認能從 GitHub 主機連上證交所。
4. 網址是 `https://<帳號>.github.io/<repo>/`。

注意：證交所可能限制來自國外雲端主機的連線。如果 Actions 抓不到資料，可以改用自己電腦上的 [self-hosted runner](https://docs.github.com/actions/hosting-your-own-runners)，把 `runs-on` 改成 `self-hosted`。

## 本機使用

```bash
uv sync
uv run python run_daily.py                       # 跟 Actions 相同的每日流程
uv run python -m http.server 8765 --directory site   # 預覽 http://localhost:8765/
```

用本機完整資料重做（需要先爬完 2005 年起的資料，約 9 小時）：

```bash
uv run python crawl_stocks.py && uv run python crawl_flows.py   # 原始資料 → data/raw/
uv run python build_dataset.py && uv run python build_flows.py  # Qlib 資料 → qlib_data/
uv run python train_horizons.py                                 # 研究版模型 → models/v1/
uv run python train_production.py --version v2                  # 正式版模型 → models/v2/
uv run python history_update.py                                 # 回填 2019 年起的排名 → state/history/
uv run python seed_state.py                                     # 近 200 個交易日原始資料 → state/raw/
uv run python build_site.py                                     # 網頁 → site/
```

## 檔案

| 路徑 | 說明 |
|---|---|
| `run_daily.py` | 每日流程 |
| `daily_crawl.py` | 增量爬蟲（每次約 3 秒一個請求） |
| `crawl_stocks.py`、`crawl_flows.py` | 完整歷史爬蟲 |
| `build_dataset.py`、`build_flows.py` | 原始資料 → Qlib 資料（路徑可用 `QTW_*` 環境變數覆寫） |
| `train_horizons.py` | 研究版（v1）訓練：保留測試期 |
| `train_production.py` | 正式版訓練：用到最新資料 |
| `history_update.py` | 新交易日排名 → `state/history/` |
| `build_site.py` | `state/history/` → `site/data.js`、`site/api/` |
| `rank_stocks.py` | 單日排名輸出 CSV |
| `tw_qlib/` | 台股交易所設定（漲跌停）、Alpha158 加法人特徵的資料處理器 |
| `models/` | `versions.json` 與各版本模型（`v1/`、`v2/`） |
| `state/raw/` | 近 200 個交易日的證交所原始 JSON |
| `state/history/` | 各版本每日分數 `scores_<版本>.csv`、0050 含息報酬 `ref_0050.csv`、股價與籌碼 `prices.csv`、加權指數 `twii.csv`、名稱 `names.csv` |
| `site/index.html` | 靜態網頁（資料由 `build_site.py` 產生） |
| `docs/EXPERIMENTS.md` | 資料建置與各輪實驗紀錄 |
| `run_experiments.py` 等 | 研究用實驗腳本 |

## 資料來源

[臺灣證券交易所](https://www.twse.com.tw/)公開資訊。資料可能有延遲或錯誤，請以官方公告為準。
