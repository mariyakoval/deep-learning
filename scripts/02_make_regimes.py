import pandas as pd
from dl_fin import config as C
from dl_fin.data.prices import add_price_features
from dl_fin.labels.hmm import fit_regimes, validate

px = add_price_features(pd.read_parquet(C.RAW / "spy_raw.parquet"))
reg = fit_regimes(px)
px.join(reg).to_parquet(C.INTERIM / "prices_regimes.parquet")
print(validate(reg).round(2))    # high-vol share should spike in GFC/COVID/2022; if not, revisit labels
