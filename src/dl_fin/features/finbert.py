import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .. import config as C

_cache = {}

def _load():
    if "m" not in _cache:
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(C.FINBERT)
        model = AutoModelForSequenceClassification.from_pretrained(C.FINBERT, output_hidden_states=True).to(dev).eval()
        _cache["m"] = (tok, model, dev)
    return _cache["m"]

@torch.no_grad()
def embed_texts(texts, bs=64, max_len=128):
    """Frozen FinBERT -> (N, 771) float32: [CLS(768) | softmax(3)], returned in input order."""
    tok, model, dev = _load()
    order = np.argsort([len(t) for t in texts])          # length-sorted batches = less padding
    out = np.zeros((len(texts), 771), dtype="float32")
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(dev)
        o = model(**enc)
        out[idx] = torch.cat([o.hidden_states[-1][:, 0], o.logits.softmax(-1)], 1).float().cpu().numpy()
    return out
