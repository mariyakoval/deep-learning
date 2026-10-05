import duckdb
import pandas as pd

con = duckdb.connect()

df = con.execute("""
    SELECT
        date,
        date_trading,
        publisher,
        stocks,
        LEFT(text, 250) AS text_preview
    FROM read_parquet('Data/sp500_news_final.parquet')
    LIMIT 10
""").fetchdf()

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

print(df.to_string(index=False))

con.close()