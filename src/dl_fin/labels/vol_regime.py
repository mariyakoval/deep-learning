"""Regime labels: bucket next-5-day realized volatility into calm / middle / crisis.
Cutoffs are quantiles of the train split only, so val/test never influence them."""
import numpy as np, pandas as pd
from .. import config as C

def cutoffs(fwd_rv: pd.Series) -> np.ndarray:
    a, b = C.SPLITS["train"]
    return fwd_rv.loc[a:b].quantile(C.REGIME_QUANTILES).values

def to_regime(vol, cuts) -> np.ndarray:
    return np.digitize(vol, cuts)      # 0=calm, 1=middle, 2=crisis
