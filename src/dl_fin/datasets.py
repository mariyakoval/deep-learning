import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from . import config as C

PRICE_MACRO = ["log_ret", "rv_21", "log_volume", "hl_range", "cpi_yoy", "t10y2y", "fedfunds"]

class MarketDataset(Dataset):
    """WINDOW-day window ending at t -> price/macro seq, news seq + mask, regime[t+1], forward vol.
    Pass train `stats` to val/test splits. news_emb=None gives the price-only baseline."""
    def __init__(self, table, news_emb=None, start=None, end=None, stats=None):
        t = table.loc[start:end]
        X = t[PRICE_MACRO].values.astype("float32")
        self.stats = stats or (X.mean(0), X.std(0) + 1e-8)
        self.X = (X - self.stats[0]) / self.stats[1]
        self.has_news = news_emb is not None
        if self.has_news:
            d = news_emb.reindex(t.index)
            self.M = (~d.isna().all(axis=1)).values.astype("float32")
            self.N = np.nan_to_num(d.values.astype("float32"))
        self.y_reg = t["next_regime"].values.astype("int64")
        self.y_vol = t["fwd_rv"].values.astype("float32")

    def __len__(self): return len(self.X) - C.WINDOW + 1

    def __getitem__(self, i):
        s, e = i, i + C.WINDOW
        item = {"x": torch.from_numpy(self.X[s:e]), "y_regime": self.y_reg[e - 1], "y_vol": self.y_vol[e - 1]}
        if self.has_news:
            item.update(news=torch.from_numpy(self.N[s:e]), news_mask=torch.from_numpy(self.M[s:e]))
        return item
