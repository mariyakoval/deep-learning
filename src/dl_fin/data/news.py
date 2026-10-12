"""Stream the HF financial-news dataset, keep S&P-500 tickers, 2005-2024.
Uniform per-day reservoir sample (MAX_ARTICLES_PER_DAY) so the stream's ticker ordering doesn't bias the sample."""
import json, random, collections
import pandas as pd
from .. import config as C

def sp500_tickers() -> set:
    cache = C.RAW / "sp500_tickers.txt"
    if cache.exists():
        return set(cache.read_text().split())
    import io, requests
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    out = set(t["Symbol"].str.replace(".", "-", regex=False).str.upper())
    C.RAW.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(out)))
    return out
    # NOTE: current constituents only (survivorship bias) - mention in the report.

def _symbols(extra) -> set:
    if not extra:
        return set()
    try:
        d = json.loads(extra) if isinstance(extra, str) else dict(extra)
    except Exception:
        return set()
    out = set()
    for k in ("stocks", "stock_symbol", "stock_symbols", "symbols", "ticker", "tickers"):
        v = d.get(k)
        if isinstance(v, str):
            out.update(x.strip().upper() for x in v.replace(";", ",").split(",") if x.strip())
        elif isinstance(v, (list, tuple)):
            out.update(str(x).upper() for x in v)
    return out

def _stream():
    from datasets import load_dataset
    return load_dataset(C.HF_DATASET, data_files=C.HF_NEWS_GLOB, split="train", streaming=True)

def inspect(n=3):
    for i, row in enumerate(_stream()):
        print({k: str(v)[:200] for k, v in row.items()})
        if i + 1 >= n:
            break

def download_news() -> pd.DataFrame:
    from tqdm import tqdm
    rng, tick = random.Random(0), sp500_tickers()
    seen, kept = collections.Counter(), collections.defaultdict(list)
    k = C.MAX_ARTICLES_PER_DAY
    for row in tqdm(_stream(), desc="news"):
        date = str(row["date"])[:10]
        if not ("2005-01-01" <= date <= "2024-12-31") or not row.get("text"):
            continue
        syms = _symbols(row.get("extra_fields")) & tick
        if not syms:
            continue
        item = (date, str(row["text"])[:2000], sorted(syms)[0])
        seen[date] += 1
        if len(kept[date]) < k:
            kept[date].append(item)
        else:
            j = rng.randrange(seen[date])        # reservoir sampling
            if j < k:
                kept[date][j] = item
    df = pd.DataFrame([r for v in kept.values() for r in v], columns=["date", "text", "ticker"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
