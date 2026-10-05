import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM
from .. import config as C

def fit_regimes(px: pd.DataFrame, seed=0) -> pd.Series:
    """Gaussian HMM on [log_ret, rv_21]. States re-ordered so 0=calm ... K-1=crisis (by mean vol)."""
    X = px[["log_ret", "rv_21"]].values
    is_tr = px.index <= C.HMM_TRAIN_END
    Z = (X - X[is_tr].mean(0)) / X[is_tr].std(0)
    m = GaussianHMM(C.N_REGIMES, covariance_type="full", n_iter=500, random_state=seed).fit(Z[is_tr])
    raw = m.predict(Z)
    remap = {s: i for i, s in enumerate(np.argsort(m.means_[:, 1]))}
    return pd.Series([remap[s] for s in raw], index=px.index, name="regime")

def validate(regime: pd.Series) -> pd.DataFrame:
    """Share of days in the highest-vol regime during known crises vs. overall."""
    hi = C.N_REGIMES - 1
    rows = {"overall": (regime == hi).mean()}
    for k, (a, b) in C.CRISES.items():
        r = regime.loc[a:b]
        rows[k] = (r == hi).mean() if len(r) else np.nan
    return pd.Series(rows, name=f"share_in_regime_{hi}").to_frame()
