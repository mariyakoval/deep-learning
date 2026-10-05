import pandas as pd
from dl_fin import config as C
from dl_fin.data.merge import build_table

px = pd.read_parquet(C.INTERIM / "prices_regimes.parquet")
macro = pd.read_parquet(C.RAW / "macro.parquet")
t = build_table(px.drop(columns="regime"), macro, px["regime"])
t.to_parquet(C.PROCESSED / "table.parquet")
print(t.shape, t.index.min().date(), "->", t.index.max().date()); print(t.regime.value_counts().sort_index())
