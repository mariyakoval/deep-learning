from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW, INTERIM, PROCESSED = (ROOT / "data" / d for d in ("raw", "interim", "processed"))

TICKER = "SPY"
START, END = "2005-01-01", "2025-01-01"          # yfinance end date is exclusive -> covers all of 2024

# FRED series -> (column name, publication lag in days to avoid look-ahead)
FRED = {"CPIAUCSL": ("cpi_yoy", 45), "T10Y2Y": ("t10y2y", 1), "FEDFUNDS": ("fedfunds", 35)}

HF_DATASET = "Brianferrell787/financial-news-multisource"   # verify with: 01_download_data.py --inspect-news
HF_NEWS_GLOB = "data/fnspid_news/*.parquet"
MAX_ARTICLES_PER_DAY = 50                                    # bounds FinBERT compute

FINBERT = "ProsusAI/finbert"

N_REGIMES = 3
HMM_TRAIN_END = "2017-12-31"     # fit HMM on train period only to limit leakage
WINDOW = 60                      # lookback days fed to the Transformer
VOL_HORIZON = 5                  # forward realized-vol horizon (days)
CRISES = {"GFC": ("2008-09-01", "2009-03-31"), "COVID": ("2020-02-20", "2020-04-30"),
          "2022_bear": ("2022-01-03", "2022-10-14"), "SVB": ("2023-03-08", "2023-03-31")}
