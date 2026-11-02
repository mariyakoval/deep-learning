"""Add price features (log return, 21-day realized vol, log volume, high-low range) to raw SPY prices."""
import pandas as pd
from dl_fin import config as C
from dl_fin.data.prices import add_price_features

px = add_price_features(pd.read_parquet(C.RAW / "spy_raw.parquet"))
px.to_parquet(C.INTERIM / "prices.parquet")
print("prices", px.shape)
