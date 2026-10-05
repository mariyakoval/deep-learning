# Predicting S&P 500 market regimes and volatility with news

## 1. The question

Can a model that reads financial news alongside price and macro data predict the S&P 500's next-day market regime (calm, middle, crisis) and its volatility over the next five trading days better than the same model using price and macro data alone?

We predict regime and volatility, not direction. Whether the market goes up or down tomorrow is close to random. How much it will swing is much more predictable, and that is what risk managers actually need.

## 2. The data

Everything is downloaded and built by the scripts in `scripts/`. The `data/` folder is git-ignored, so run the scripts to recreate it.

| Source | What | Link |
|---|---|---|
| yfinance | SPY daily prices and volume, 2005 to 2024 | https://pypi.org/project/yfinance/ |
| FRED | CPI (inflation), 10Y-2Y Treasury spread, federal funds rate | https://fred.stlouisfed.org/series/CPIAUCSL, https://fred.stlouisfed.org/series/T10Y2Y, https://fred.stlouisfed.org/series/FEDFUNDS |
| Hugging Face | `fnspid_news` subset of financial-news-multisource | https://huggingface.co/datasets/Brianferrell787/financial-news-multisource |

The news dataset is gated: create a Hugging Face account, accept the terms on the dataset page, and run `huggingface-cli login`.

**One row** in the final table is one trading day: 5,007 days from 2005-02-02 to 2024-12-23. Features are log return, 21-day realized volatility, log volume, daily high-low range, CPI year-over-year, the 10Y-2Y spread, and the fed funds rate.

**News.** We scanned 28.6 million rows and kept 237,280 articles about S&P 500 companies, at most 50 per day. Each article is embedded with a frozen FinBERT model and averaged per trading day. News exists from 2009-05 to 2023-12, and 98.5% of trading days in that window have at least one article. Articles from weekends and holidays are assigned to the next trading day so the model never sees news early.

**Labels.** There are no human labels. We made them:

- **Regime:** a 3-state Gaussian hidden Markov model on returns and realized volatility, fit on data through 2017 only. States are ordered by volatility: calm (2,231 days, 45%), middle (2,110 days, 42%), crisis (666 days, 13%). The target is the next day's regime.
- **Volatility:** realized volatility over the next five trading days, annualized.

**Split.** Train 2009-05 to 2017, validation 2018 to 2019, test 2020 to 2023. The test period contains COVID, the 2022 bear market and the SVB collapse, none of which the model sees in training.

## 3. Who cares

Anyone who sizes positions or hedges: risk managers, portfolio managers, and options traders. A good regime and volatility forecast tells them when to cut exposure or buy protection.

The two mistakes are not equal. **A miss is worse than a false alarm.** Calling a crisis "calm" leaves someone under-hedged in a crash. Calling "calm" a crisis costs some hedging fees and lost upside. We therefore report recall on the crisis regime, not just accuracy.

Ethics: the data is public and contains no personal information. The main risk is misuse. A backtested model is not investment advice, and a missed crash call can cost real money. News comes from commercial publishers, so check the dataset's license before any use beyond coursework.

## 4. The baseline

Simplest things to beat:

- **Majority class:** always predict "calm". This is right on 44.6% of days across the full sample. We will recompute it on the test window.
- **Persistence:** predict that tomorrow's regime equals today's, and that the next five days' volatility equals today's 21-day realized volatility. Numbers to be added in week 3.
- **Main baseline:** the same Transformer with the same heads, trained on the same splits, but with no news input. The difference between this model and ours is the value of the news.

Accuracy alone is misleading here, since "never predict crisis" is already 87% accurate. We also report crisis-regime recall and F1, and volatility error (MAE).

## 5. The plan

Names to fill in before submitting.

| Weeks | Work | Who |
|---|---|---|
| 1-2 | Data pipeline: prices, macro, news, FinBERT embeddings, HMM regimes. **Done.** | [name] |
| 3 | Persistence baselines. Price-only Transformer with regime head. | [name] |
| 4-5 | News branch: Transformer encoder with cross-attention over daily news embeddings. | [name] |
| 6 | First full training run on the news model. Go/no-go check on whether the data is enough. | [name], [name] |
| 7-8 | Add the volatility head. Tune the combined loss and class weights. | [name] |
| 9-10 | Full test-set evaluation against all baselines, several random seeds. | [name] |
| 11-12 | Event analysis: COVID, the 2022 bear market, SVB. Ablations (news embeddings vs sentiment scores only). | [name], [name] |
| 13 | Figures and write-up. | everyone |
| 14 | Final presentation and buffer. | everyone |

## 6. The risk

- **Not enough data for a Transformer.** Training covers about 2,200 trading days. We keep the model small (2 to 4 layers) and compare against simpler baselines. By week 6 we will have trained once and know whether it learns anything.
- **Self-made labels.** Results are only as good as the HMM. We checked it against known crises: the crisis state covers 94% of days in the 2008 crisis, 96% of COVID, and 52% of the 2022 bear market. It covers 0% of SVB, because SPY's own volatility stayed low in March 2023. A news advantage on SVB may therefore show up in the volatility head, not the regime head.
- **Rare crisis class.** Only 13% of days are crisis. Use class weights and watch crisis recall from the first training run.
- **Noisy news.** Many articles are generic or only loosely tied to the market. Week 11 ablations will show whether the news embeddings add anything.
- **Survivorship bias.** News is filtered to today's S&P 500 members, not the index as it was at the time.
- **Leakage.** The HMM is fit on pre-2018 data, and lookback windows must not reach across split boundaries. Check this before reporting any test number.

## Run it

```bash
pip install -r requirements.txt && pip install -e .
python scripts/01_download_data.py
python scripts/02_make_regimes.py
python scripts/04_build_dataset.py
python scripts/03_embed_news.py
```

Outputs land in `data/processed/` (`table.parquet` and `news_emb.parquet`).
