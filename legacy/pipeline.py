import pandas as pd
import yfinance as yf
from fredapi import Fred
import os
FRED_API_KEY = os.getenv("FRED_API_KEY")

fred = Fred(api_key=FRED_API_KEY)

print("Downloading SPY data from yfinance")

spy = yf.download("SPY", start="2005-01-01", progress=False)

if isinstance(spy.columns, pd.MultiIndex): # Clean multi-index columns
  spy.columns = spy.columns.get_level_values(0)
spy["Return"] = spy["Close"].pct_change() # Calculate daily simple returns

# Target SPY columns
spy_df = pd.DataFrame(
    {
        "Price": spy["Close"],
        "Volume": spy["Volume"],
        "Return": spy["Return"],
    }
)
print("Pulling macroeconomic data from FRED")

# CPIAUCSL = Consumer Price Index for All Urban Consumers (Monthly)
# T10Y2Y   = 10-Year Minus 2-Year Treasury Yield Spread (Daily)
# FEDFUNDS = Effective Federal Funds Rate (Monthly)
cpi = fred.get_series("CPIAUCSL", observation_start="2005-01-01")
spread = fred.get_series("T10Y2Y", observation_start="2005-01-01")
fed_funds = fred.get_series("FEDFUNDS", observation_start="2005-01-01")

macro_df = pd.DataFrame(
    {"CPI": cpi, "Spread_10Y_2Y": spread, "Fed_Funds_Rate": fed_funds}
)

macro_aligned = macro_df.reindex(spy_df.index).ffill()
final_df = pd.concat([spy_df, macro_aligned], axis=1)
final_df = final_df.dropna() # Drop any initial rows containing missing values 

print("\n--- Merged Data Preview ---")
print(final_df.tail())
output_dir = "Data"
os.makedirs(
    output_dir, exist_ok=True
) 

output_path = os.path.join(output_dir, "spy_macro_merged.csv")
final_df.to_csv(output_path)
print(f"\nSaved merged dataset to {output_path}")