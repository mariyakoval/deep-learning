"""Download prices, FRED macro, filtered news. Run with --inspect-news first to verify HF column names.
Flags: --skip-news"""
import sys
from dl_fin import config as C
from dl_fin.data import prices, macro, news

if "--inspect-news" in sys.argv:
    news.inspect(); sys.exit()
for d in (C.RAW, C.INTERIM, C.PROCESSED): d.mkdir(parents=True, exist_ok=True)
px = prices.download_prices();       px.to_parquet(C.RAW / "spy_raw.parquet");  print("prices", px.shape)
mc = macro.download_macro(px.index); mc.to_parquet(C.RAW / "macro.parquet");    print("macro", mc.shape)
if "--skip-news" not in sys.argv:
    nw = news.download_news();       nw.to_parquet(C.RAW / "news.parquet");     print("news", nw.shape)
