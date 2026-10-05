import os
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# How many rows to process at once. Lower this if you still hit memory issues.
BATCH_SIZE = 50_000

# Possible column names that might hold ticker symbols in your
# sp500_constituents.csv — the script checks these in order.
TICKER_COLUMN_CANDIDATES = ["Ticker", "Symbol", "symbol", "ticker"]

# Project root = two levels up from this script
# (assumes this file lives directly inside a top-level "scripts/" folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")

final_path = os.path.join(DATA_DIR, "sp500_news_filtered.parquet")
tickers_csv_path = os.path.join(DATA_DIR, "sp500_constituents.csv")

if not os.path.exists(final_path):
    raise FileNotFoundError(
        f"Could not find {final_path}. "
        "Run the download+filter script first to produce this file."
    )

if not os.path.exists(tickers_csv_path):
    raise FileNotFoundError(
        f"Could not find {tickers_csv_path}. "
        "Make sure sp500_constituents.csv is in the Data/ folder."
    )

print("Loading S&P 500 ticker list...")
constituents = pd.read_csv(tickers_csv_path)
constituents.columns = constituents.columns.str.strip()  # guard against stray whitespace in headers

ticker_column = None
for candidate in TICKER_COLUMN_CANDIDATES:
    if candidate in constituents.columns:
        ticker_column = candidate
        break

if ticker_column is None:
    raise ValueError(
        f"Couldn't find a ticker column in sp500_constituents.csv. "
        f"Columns found: {list(constituents.columns)}. "
        f"Add the actual column name to TICKER_COLUMN_CANDIDATES at the top of this script."
    )

sp500_tickers = set(constituents[ticker_column].astype(str).str.strip().str.upper())
print(f"Loaded {len(sp500_tickers)} S&P 500 tickers from column '{ticker_column}'.")

print("\nFiltering news to S&P 500-relevant articles (processed in batches to limit memory use)...")


def has_sp500_ticker(stocks_value):
    if stocks_value is None:
        return False
    if isinstance(stocks_value, str):
        try:
            stocks_value = json.loads(stocks_value)
        except (json.JSONDecodeError, TypeError):
            return False
    try:
        return any(str(s).strip().upper() in sp500_tickers for s in stocks_value)
    except TypeError:
        return False


sp500_final_path = os.path.join(DATA_DIR, "sp500_news_final.parquet")

parquet_file = pq.ParquetFile(final_path)
writer = None
total_in = 0
total_out = 0

for batch_num, record_batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE)):
    chunk = record_batch.to_pandas()
    total_in += len(chunk)

    chunk["is_sp500"] = chunk["stocks"].apply(has_sp500_ticker)
    filtered = chunk[chunk["is_sp500"]].drop(columns=["is_sp500"])
    total_out += len(filtered)

    if len(filtered) > 0:
        table = pa.Table.from_pandas(filtered, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(sp500_final_path, table.schema)
        writer.write_table(table)

    if (batch_num + 1) % 10 == 0:
        print(f"  ...processed {total_in:,} rows so far, kept {total_out:,}")

if writer is not None:
    writer.close()

print("\n----------------------------------------")
print("S&P 500 ticker filtering complete.")
print("----------------------------------------")
print(f"Final dataset: {sp500_final_path}")
print(f"S&P 500-relevant articles: {total_out:,} of {total_in:,} stock-tagged articles")
print("----------------------------------------")