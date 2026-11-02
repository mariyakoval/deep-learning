# Predicting S&P 500 market regimes and volatility with news

## 1. The question

Can a model that reads financial news alongside price and macro data predict the S&P 500's volatility regime (calm, middle, crisis) and its realized volatility over the next five trading days better than the same model using price and macro data alone?

We predict regime and volatility, not direction. Whether the market goes up or down tomorrow is close to random. How much it will swing is much more predictable, and that is what risk managers actually need.

## 2. The data

Everything is downloaded and built by the scripts in `scripts/`. The `data/` folder is git-ignored except `data/processed/table.parquet`, a snapshot of the final table so results reproduce even if yfinance or FRED revise their data.

| Source | What | Link |
|---|---|---|
| yfinance | SPY daily prices and volume, 2005 to 2024 | https://pypi.org/project/yfinance/ |
| FRED | CPI (inflation), 10Y-2Y Treasury spread, federal funds rate | https://fred.stlouisfed.org/series/CPIAUCSL, https://fred.stlouisfed.org/series/T10Y2Y, https://fred.stlouisfed.org/series/FEDFUNDS |
| Hugging Face | `fnspid_news` subset of financial-news-multisource | https://huggingface.co/datasets/Brianferrell787/financial-news-multisource |

The news dataset is gated: create a Hugging Face account, accept the terms on the dataset page, and run `huggingface-cli login`.

**One row** in the final table is one trading day: 4,773 days from 2006-01-06 to 2024-12-23 (2005 is used up by the 1-year rolling volume feature). Features are log return, 21-day realized volatility, log volume, volume relative to its trailing 1-year mean, daily high-low range, CPI year-over-year, the 10Y-2Y spread, and the fed funds rate.

**News.** We scanned 28.6 million rows and kept 237,280 articles about S&P 500 companies, at most 50 per day. Each article is embedded with a frozen FinBERT model and averaged per trading day. News exists from 2009-05 to 2023-12, and 98.5% of trading days in that window have at least one article. Articles from weekends and holidays are assigned to the next trading day so the model never sees news early.

**Labels.** There are no human labels. We made them:

- **Volatility:** realized volatility over the next five trading days (t+1 to t+5), annualized.
- **Regime:** that forward volatility bucketed at the 50th and 87th percentiles of the train split (10.6% and 21.3% annualized): calm, middle, crisis. On the test window this gives 29% calm, 45% middle, 26% crisis.

We first used a hidden Markov model for the regimes, but dropped it: its labels were 97% predictable from "tomorrow = today", and Viterbi decoding uses future data. The old code is in `legacy/hmm.py`.

**Split.** Train 2009-05 to 2017, validation 2018 to 2019, test 2020 to 2023 (`config.SPLITS`). The last 5 days of each split are dropped so no target overlaps the next split. The test period contains COVID, the 2022 bear market and the SVB collapse, none of which the model sees in training.

## 3. Who cares

Anyone who sizes positions or hedges: risk managers, portfolio managers, and options traders. A good regime and volatility forecast tells them when to cut exposure or buy protection.

The two mistakes are not equal. **A miss is worse than a false alarm.** Calling a crisis "calm" leaves someone under-hedged in a crash. Calling "calm" a crisis costs some hedging fees and lost upside. We therefore report recall on the crisis regime, not just accuracy.

Ethics: the data is public and contains no personal information. The main risk is misuse. A backtested model is not investment advice, and a missed crash call can cost real money. News comes from commercial publishers, so check the dataset's license before any use beyond coursework.

## 4. The baseline

Simplest things to beat:

- **Majority class:** always predict "calm" and the train median volatility.
- **Persistence:** the next five days look like the last five: predict the trailing 5-day realized volatility and its regime.
- **Main baseline:** the same network trained on the same splits, but with no news input. The difference between this model and ours is the value of the news.

Accuracy alone is misleading with a rare crisis class, so we report macro-F1, crisis-regime recall, and volatility MAE/RMSE.

### Milestone 1 results (test 2020-2023, 1,001 days)

| Model | Params | Macro-F1 | Crisis recall | Accuracy | Vol MAE | Vol RMSE |
|---|---|---|---|---|---|---|
| Majority | 0 | 0.151 | 0.000 | 0.292 | 0.093 | 0.165 |
| Persistence | 0 | 0.543 | 0.574 | 0.541 | 0.077 | 0.116 |
| 1D CNN (5 seeds) | 4,356 | 0.587 ± 0.015 | 0.765 ± 0.023 | 0.581 ± 0.015 | 0.069 ± 0.003 | 0.129 ± 0.009 |

The CNN sees the last 20 days of log return, 21-day vol, relative volume and high-low range. Window, channels, learning rate and feature set were picked on validation loss only (`results/hparam_search.csv`). Macro features were dropped by that search: fed funds in the test years sits up to 19 train standard deviations above its training mean. Everything is produced by `notebooks/milestone1.ipynb`.

## 5. The plan

| Weeks | Work | Who |
|---|---|---|
| 1-2 | Data pipeline: prices, macro, news, FinBERT embeddings, regime labels. **Done.** | Prisha & Mariia (Initial Pass) |
| 3 | Persistence baselines. Price-only 1D CNN with regime and volatility heads (Milestone 1). **Done.** | Prisha (Initial Pass) & Mariia |
| 4-5 | News branch: Transformer encoder with cross-attention over daily news embeddings. | Prisha & Mariia (Initial Pass) |
| 6 | First full training run on the news model. Go/no-go check on whether the data is enough. | both |
| 7-8 | Add the volatility head. Tune the combined loss and class weights. | Prisha (Initial Pass) & Mariia  |
| 9-10 | Full test-set evaluation against all baselines, several random seeds. | Prisha & Mariia (Initial Pass) |
| 11-12 | Event analysis: COVID, the 2022 bear market, SVB. Ablations (news embeddings vs sentiment scores only). | both |
| 13 | Figures and write-up. | both |
| 14 | Final presentation and buffer. | both |

## 6. The risk

- **Not enough data for a Transformer.** Training covers about 2,200 trading days. We keep the model small (2 to 4 layers) and compare against simpler baselines. By week 6 we will have trained once and know whether it learns anything.
- **Self-made labels.** Regimes are fixed volatility bands, so they are only as meaningful as the cutoffs. SVB has only 3 crisis days because SPY's own volatility stayed low in March 2023, so a news advantage there may show up in the volatility head, not the regime head.
- **Rare crisis class.** 13% of train days are crisis by construction. Use class weights and watch crisis recall from the first training run.
- **Noisy news.** Many articles are generic or only loosely tied to the market. Week 11 ablations will show whether the news embeddings add anything.
- **Survivorship bias.** News is filtered to today's S&P 500 members, not the index as it was at the time.
- **Leakage.** Regime cutoffs and feature normalization use the train split only, and targets never cross a split boundary. Input windows may reach back before a split, since past features are known at prediction time.

## Run it

Python 3.9, CPU only. Install, then run the Milestone 1 notebook (about 5 minutes; it uses the committed table, so no downloads):

```bash
pip install -r requirements.txt && pip install -e .
jupyter nbconvert --to notebook --execute --inplace notebooks/milestone1.ipynb
```

Seeds are fixed and PyTorch runs in deterministic mode; two clean runs give byte-identical `results/*.csv`. Figures go to `figures/`.

To rebuild the data from scratch (needs internet and Hugging Face access for news):

```bash
python scripts/01_download_data.py
python scripts/02_make_features.py
python scripts/03_build_dataset.py
python scripts/04_embed_news.py
```

Outputs land in `data/processed/` (`table.parquet` and `news_emb.parquet`).
