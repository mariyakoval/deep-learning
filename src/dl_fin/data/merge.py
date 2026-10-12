import numpy as np, pandas as pd
from .. import config as C

def build_table(px, macro, regime) -> pd.DataFrame:
    df = px.join(macro, how="left").join(regime, how="left")
    # targets: realized vol over the NEXT VOL_HORIZON days; next day's regime
    df["fwd_rv"] = df.log_ret[::-1].rolling(C.VOL_HORIZON).std()[::-1].shift(-1) * np.sqrt(252)
    df["next_regime"] = df["regime"].shift(-1)
    return df.dropna()
