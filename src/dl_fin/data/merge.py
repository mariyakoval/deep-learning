"""Join price features and macro data, then add the forward-volatility target and regime labels."""
import numpy as np, pandas as pd
from .. import config as C
from ..labels.vol_regime import cutoffs, to_regime

def build_table(px, macro) -> pd.DataFrame:
    df = px.join(macro, how="left")
    df["rv_5"] = df.log_ret.rolling(C.VOL_HORIZON).std() * np.sqrt(252)   # trailing 5-day vol, used by the persistence baseline
    df["fwd_rv"] = df.rv_5.shift(-C.VOL_HORIZON)                           # target: vol over days t+1 .. t+5
    df = df.dropna()
    df["regime"] = to_regime(df.fwd_rv, cutoffs(df.fwd_rv))                # regime of the next 5 days
    return df
