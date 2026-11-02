"""PyTorch dataset: a window of past price/macro features -> regime and log volatility of the next 5 days."""
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from . import config as C

PRICE_MACRO = ["log_ret", "rv_21", "log_volume", "hl_range", "cpi_yoy", "t10y2y", "fedfunds"]
FEATURE_SETS = {"price+macro": PRICE_MACRO,
                "price": ["log_ret", "rv_21", "log_volume", "hl_range"],
                "price_relvol": ["log_ret", "rv_21", "rel_volume", "hl_range"]}

def split_days(table, split) -> pd.DatetimeIndex:
    """Target days of a split. The last VOL_HORIZON days are dropped so no target reaches into the next split."""
    a, b = C.SPLITS[split]
    return table.loc[a:b].index[:-C.VOL_HORIZON]

def fit_stats(table, features=PRICE_MACRO) -> dict:
    """Normalisation constants for features and log-vol target, from the train split only."""
    t = table.loc[split_days(table, "train")]
    x, v = t[features].values, np.log(t.fwd_rv.values)
    return {"features": list(features), "x_mean": x.mean(0), "x_std": x.std(0), "v_mean": v.mean(), "v_std": v.std()}

class MarketDataset(Dataset):
    """Sample = the `window` days ending at a target day t. A window may start before the split
    (past features are known at t); only the target days are restricted to the split.
    news_emb=None gives the price-only model."""
    def __init__(self, table, split, stats, window=C.WINDOW, news_emb=None):
        self.window = window
        self.dates = split_days(table, split)
        self.end = table.index.get_indexer(self.dates)       # row position of each target day
        assert self.end.min() >= window - 1, "not enough history before the split"
        self.X = ((table[stats["features"]].values - stats["x_mean"]) / stats["x_std"]).astype("float32")
        self.y_reg = table.regime.values.astype("int64")
        self.y_vol = ((np.log(table.fwd_rv.values) - stats["v_mean"]) / stats["v_std"]).astype("float32")
        self.has_news = news_emb is not None
        if self.has_news:
            d = news_emb.reindex(table.index)
            self.M = (~d.isna().all(axis=1)).values.astype("float32")   # 1 if the day has news
            self.N = np.nan_to_num(d.values.astype("float32"))

    def __len__(self): return len(self.end)

    def __getitem__(self, i):
        e = self.end[i] + 1
        s = e - self.window
        item = {"x": torch.from_numpy(self.X[s:e]), "y_regime": self.y_reg[e - 1], "y_vol": self.y_vol[e - 1]}
        if self.has_news:
            item.update(news=torch.from_numpy(self.N[s:e]), news_mask=torch.from_numpy(self.M[s:e]))
        return item
