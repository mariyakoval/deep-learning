"""Join prices and macro, add the forward-vol target and regime labels -> data/processed/table.parquet."""
import pandas as pd
from dl_fin import config as C
from dl_fin.data.merge import build_table
from dl_fin.labels.vol_regime import cutoffs

px = pd.read_parquet(C.INTERIM / "prices.parquet")
macro = pd.read_parquet(C.RAW / "macro.parquet")
t = build_table(px, macro)
t.to_parquet(C.PROCESSED / "table.parquet")
print(t.shape, t.index.min().date(), "->", t.index.max().date())
print("regime cutoffs (annualized vol):", cutoffs(t.fwd_rv).round(4))
print(t.regime.value_counts().sort_index())
