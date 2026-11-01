import os
import glob
import json
import shutil
import time
 
import duckdb
import pandas as pd
from huggingface_hub import snapshot_download


 
# 0. SETTINGS
CLEANUP_AFTER_SUCCESS = True


# 1. SET UP DIRECTORIES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "fnspid_news_raw")
CHUNKS_DIR = os.path.join(DATA_DIR, "raw", "filtered_chunks")
DUCKDB_TMP_DIR = os.path.join(DATA_DIR, "raw", "duckdb_tmp")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(DUCKDB_TMP_DIR, exist_ok=True)

# 2. DOWNLOAD RAW DATA FROM HUGGING FACE
print("Downloading financial news dataset from Hugging Face...")

snapshot_download(
    repo_id="Brianferrell787/financial-news-multisource",
    repo_type="dataset",
    allow_patterns=["data/fnspid_news/*.parquet"],
    local_dir=RAW_DIR
)

print("Download complete.")

# 3. FIND RAW PARQUET FILES

files = sorted(
    glob.glob(
        os.path.join(RAW_DIR, "data", "fnspid_news", "fnspid_news.*.parquet")
    )
)

print(f"Found {len(files)} Parquet files to process.")

if not files:
    raise FileNotFoundError(
        "No Parquet files were found. "
    )
 
# 4. CONNECT TO DUCKDB
 
con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'")
con.execute(f"PRAGMA temp_directory='{DUCKDB_TMP_DIR}'")
 
# 5. FILTER EACH PARQUET FILE (articles with any stock tag)
 
print("\nFiltering articles with stock metadata...\n")

for i, file_path in enumerate(files):
    output_path = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}.parquet")

    if os.path.exists(output_path):
        print(f"[{i + 1}/{len(files)}] Skipping {os.path.basename(file_path)}, already processed.")
        continue

    start_time = time.time()
    print(f"[{i + 1}/{len(files)}] Processing {os.path.basename(file_path)}...")

    con.execute(
        f"""
        COPY (
            SELECT
                date,
                text,
                json_extract_string(extra_fields, '$.date_trading') AS date_trading,
                json_extract_string(extra_fields, '$.publisher') AS publisher,
                json_extract(extra_fields, '$.stocks') AS stocks
            FROM read_parquet('{file_path}')
            WHERE json_extract(extra_fields, '$.stocks') IS NOT NULL
        )
        TO '{output_path}'
        (FORMAT PARQUET)
        """
    )

    elapsed = time.time() - start_time
    print(f"    Done in {elapsed:.1f} seconds.")

 
# 6. COMBINE FILTERED CHUNKS
 
print("\nCombining filtered chunks...")

final_path = os.path.join(DATA_DIR, "sp500_news_filtered.parquet")

con.execute(
    f"""
    COPY (
        SELECT * FROM read_parquet('{CHUNKS_DIR}/*.parquet')
    )
    TO '{final_path}'
    (FORMAT PARQUET)
    """
)

# 7. CHECK FINAL DATASET
 
count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{final_path}')").fetchone()[0]

print("\n----------------------------------------")
print("News processing complete.")
print("----------------------------------------")
print(f"Final dataset: {final_path}")
print(f"Number of articles: {count:,}")
print("----------------------------------------")

con.close()

# 8. NARROW DOWN TO S&P 500 TICKERS ONLY
TICKER_COLUMN_CANDIDATES = ["Ticker", "Symbol", "symbol", "ticker"]

print("\nLoading S&P 500 ticker list...")
 
tickers_csv_path = os.path.join(DATA_DIR, "sp500_constituents.csv")
 
if not os.path.exists(tickers_csv_path):
    raise FileNotFoundError(
        f"Could not find {tickers_csv_path}. "
        "Make sure sp500_constituents.csv is in the Data/ folder."
    )
 
constituents = pd.read_csv(tickers_csv_path)
 
ticker_column = None
for candidate in TICKER_COLUMN_CANDIDATES:
    if candidate in constituents.columns:
        ticker_column = candidate
        break
 
if ticker_column is None:
    raise ValueError(
        f"Couldn't find a ticker column in sp500_constituents.csv. "
        f"Columns found: {list(constituents.columns)}. "
    )
 
sp500_tickers = set(constituents[ticker_column].astype(str).str.strip().str.upper())
print(f"Loaded {len(sp500_tickers)} S&P 500 tickers from column '{ticker_column}'.")
 
print("\nFiltering news to S&P 500-relevant articles...")
 
news_df = pd.read_parquet(final_path)
 
 
def has_sp500_ticker(stocks_value):
    if stocks_value is None:
        return False
    # stocks_value may already be a list/array (from Parquet), or a JSON string
    if isinstance(stocks_value, str):
        try:
            stocks_value = json.loads(stocks_value)
        except (json.JSONDecodeError, TypeError):
            return False
    try:
        return any(str(s).strip().upper() in sp500_tickers for s in stocks_value)
    except TypeError:
        return False
 
 
news_df["is_sp500"] = news_df["stocks"].apply(has_sp500_ticker)
sp500_news = news_df[news_df["is_sp500"]].drop(columns=["is_sp500"])
 
sp500_final_path = os.path.join(DATA_DIR, "sp500_news_final.parquet")
sp500_news.to_parquet(sp500_final_path, index=False)
 
print("\n----------------------------------------")
print("S&P 500 ticker filtering complete.")
print("----------------------------------------")
print(f"Final dataset: {sp500_final_path}")
print(f"S&P 500-relevant articles: {len(sp500_news):,} of {len(news_df):,} stock-tagged articles")
print("----------------------------------------")
 
# 8. CLEANUP
 
if CLEANUP_AFTER_SUCCESS:
    print("\nCleaning up intermediate files (raw download + chunks)...")
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
    shutil.rmtree(DUCKDB_TMP_DIR, ignore_errors=True)
    print("Done.")