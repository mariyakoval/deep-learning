"""Seeding, training loop with early stopping, and prediction for the CNN."""
import copy, random
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .datasets import MarketDataset
from .models import CNN

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def class_weights(ds) -> torch.Tensor:
    """Inverse class frequency on the train target days, so crisis errors cost more."""
    counts = np.bincount(ds.y_reg[ds.end], minlength=3)
    return torch.tensor(len(ds) / (3 * counts), dtype=torch.float32)

def run_epoch(model, loader, w, opt=None) -> float:
    """One pass over the loader; trains if opt is given. Loss = weighted cross-entropy + MSE on standardized log vol."""
    model.train(opt is not None)
    total = 0.0
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            logits, v = model(b["x"])
            loss = F.cross_entropy(logits, b["y_regime"], weight=w) + F.mse_loss(v, b["y_vol"])
            if opt is not None:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b["x"])
    return total / len(loader.dataset)

def fit(table, stats, window=60, ch=32, lr=1e-3, seed=0, epochs=100, patience=10, batch_size=64):
    """Train on the train split, early-stop on val loss, return (best model, per-epoch loss history)."""
    set_seed(seed)
    tr, va = MarketDataset(table, "train", stats, window), MarketDataset(table, "val", stats, window)
    model = CNN(n_feat=len(stats["features"]), window=window, ch=ch)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_loader = DataLoader(tr, batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    va_loader = DataLoader(va, 256)
    w = class_weights(tr)
    hist, best, best_ep = [], float("inf"), 0
    for ep in range(epochs):
        hist.append((run_epoch(model, tr_loader, w, opt), run_epoch(model, va_loader, w)))
        if hist[-1][1] < best:
            best, best_ep, best_state = hist[-1][1], ep, copy.deepcopy(model.state_dict())
        elif ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist, columns=["train_loss", "val_loss"])

@torch.no_grad()
def predict(model, table, stats, split, window):
    """Regime class and annualized vol for every target day of a split."""
    ds = MarketDataset(table, split, stats, window)
    model.eval()
    logits, v = model(torch.stack([ds[i]["x"] for i in range(len(ds))]))
    return logits.argmax(1).numpy(), np.exp(v.numpy() * stats["v_std"] + stats["v_mean"])
