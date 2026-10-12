import pandas as pd
from .. import config as C

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"   # no API key needed

def _fetch(series: str) -> pd.Series:
    df = pd.read_csv(URL.format(series))
    d, v = df.columns[0], df.columns[1]
    s = pd.Series(pd.to_numeric(df[v], errors="coerce").values, index=pd.to_datetime(df[d]), name=series)
    return s.dropna()

def download_macro(trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for sid, (name, lag) in C.FRED.items():
        s = _fetch(sid)
        if sid == "CPIAUCSL":
            s = s.pct_change(12) * 100                 # YoY inflation
        s = s.dropna()
        s.index = s.index + pd.Timedelta(days=lag)     # shift to approx. release date
        s = s[~s.index.duplicated()]
        cols[name] = s.reindex(s.index.union(trading_days)).ffill().reindex(trading_days)
    out = pd.DataFrame(cols)
    out.index.name = "date"
    return out
