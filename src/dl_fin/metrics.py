"""Error metrics: regime classification and volatility regression."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

def score(y_reg, p_reg, y_vol, p_vol) -> dict:
    return {"macro_f1": f1_score(y_reg, p_reg, average="macro", labels=[0, 1, 2], zero_division=0),
            "crisis_recall": recall_score(y_reg, p_reg, labels=[2], average="macro", zero_division=0),
            "accuracy": accuracy_score(y_reg, p_reg),
            "vol_mae": np.mean(np.abs(y_vol - p_vol)),
            "vol_rmse": np.sqrt(np.mean((y_vol - p_vol) ** 2))}
