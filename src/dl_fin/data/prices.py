import numpy as np, pandas as pd, yfinance as yf
from .. import config as C

def download_prices() -> pd.DataFrame:
    df = yf.Ticker(C.TICKER).history(start=C.START, end=C.END, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df[["Open", "High", "Low", "Close", "Volume"]]

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    o["log_ret"] = np.log(o.Close).diff()
    o["rv_21"] = o.log_ret.rolling(21).std() * np.sqrt(252)
    o["log_volume"] = np.log(o.Volume.replace(0, np.nan))
    o["hl_range"] = np.log(o.High / o.Low)
    return o.dropna()
