"""Resumable FinBERT embedding. If news.parquet is ever regenerated, delete data/interim/news_emb_parts/ first."""
import numpy as np, pandas as pd
from tqdm import tqdm
from dl_fin import config as C
from dl_fin.features.finbert import embed_texts

CHUNK = 10_000
news = pd.read_parquet(C.RAW / "news.parquet")
td = pd.read_parquet(C.INTERIM / "prices_regimes.parquet").index

# non-trading-day articles (weekends/holidays) -> NEXT trading day (no look-ahead)
pos = td.searchsorted(news["date"].values)
keep = pos < len(td)
news, pos = news[keep].reset_index(drop=True), pos[keep]
news["date"] = td[pos]

parts = C.INTERIM / "news_emb_parts"; parts.mkdir(parents=True, exist_ok=True)
embs = []
for k, s in enumerate(tqdm(range(0, len(news), CHUNK), desc="FinBERT chunks")):
    f = parts / f"chunk_{k:04d}.npy"
    if not f.exists():
        np.save(f, embed_texts(news["text"].iloc[s:s + CHUNK].tolist()))
    embs.append(np.load(f))

df = pd.DataFrame(np.concatenate(embs), index=pd.DatetimeIndex(news["date"]))
df.columns = df.columns.astype(str)
g = df.groupby(level=0)
emb, cnt = g.mean().sort_index(), g.size().rename("news_count").sort_index()
emb.to_parquet(C.PROCESSED / "news_emb.parquet"); cnt.to_frame().to_parquet(C.PROCESSED / "news_count.parquet")
print(emb.shape, "trading days with news; median articles/day:", cnt.median())
