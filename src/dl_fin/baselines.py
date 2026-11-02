"""Zero-parameter baselines. Each returns (regime prediction, annualized vol prediction) for the given days."""
import numpy as np
from .datasets import split_days
from .labels.vol_regime import cutoffs, to_regime

def majority(table, dates):
    """Most common train regime and the train median of forward vol, every day."""
    tr = table.loc[split_days(table, "train")]
    n = len(dates)
    return np.full(n, tr.regime.mode()[0]), np.full(n, tr.fwd_rv.median())

def persistence(table, dates):
    """Next 5 days look like the last 5 days: trailing 5-day vol and its regime."""
    rv = table.loc[dates, "rv_5"].values
    return to_regime(rv, cutoffs(table.fwd_rv)), rv
