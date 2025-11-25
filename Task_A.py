import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# --- Task 1 & 2: Data Extraction ---
ticker = "TATAMOTORS.NS"
print(f"Downloading data for {ticker}...")

# Fix for FutureWarning: Explicitly set auto_adjust=False or True depending on preference. 
# We use auto_adjust=True to get the actual traded price (adjusting for splits/dividends).
stock_data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)

# Data Cleaning: Ensure we have a clean Series
# If it's a MultiIndex (common in new yfinance), drop the ticker level
if isinstance(stock_data.columns, pd.MultiIndex):
    try:
        # Try fetching 'Close'
        stock_data = stock_data['Close']
    except KeyError:
        # If 'Close' isn't there (due to auto_adjust), use the first column which is usually price
        stock_data = stock_data.iloc[:, 0]
elif 'Close' in stock_data.columns:
     stock_data = stock_data['Close']
else:
    # Fallback: take the first column available
    stock_data = stock_data.iloc[:, 0]

# Force it to be a 1D Series and drop NaNs
stock_data = stock_data.squeeze().dropna()

# --- Task 3: Compute Statistics ---
# 1. Daily Log Returns: ln(Pt / Pt-1)
log_returns = np.log(stock_data / stock_data.shift(1)).dropna()

# 2. Annualized Volatility
daily_std = log_returns.std()
annualized_vol = daily_std * np.sqrt(252)

# 3. Skewness and Kurtosis
# We convert inputs to a flat numpy array to ensure scipy returns a scalar, not an array
ret_skew = skew(log_returns.values)
ret_kurt = kurtosis(log_returns.values, fisher=True)

# Create a Summary Table
# We use .item() if the result is still a numpy object, just in case
summary_stats = pd.DataFrame({
    "Statistic": ["Last Price (ATM)", "Annualized Volatility", "Skewness", "Kurtosis", "Observations"],
    "Value": [
        round(float(stock_data.iloc[-1]), 2),
        round(float(annualized_vol), 4),
        round(float(ret_skew), 4),  # Cast to float to fix the TypeError
        round(float(ret_kurt), 4),  # Cast to float to fix the TypeError
        len(log_returns)
    ]
})

print("\n--- Part A: Summary Statistics ---")
print(summary_stats)

# --- Deliverable: Plotting ---
plt.figure(figsize=(10, 6))
sns.histplot(log_returns, kde=True, bins=30, color='blue', stat='density')
plt.title(f"Log Returns Distribution: {ticker} (Last 3 Months)")
plt.xlabel("Daily Log Returns")
plt.ylabel("Density")

plt.axvline(x=log_returns.mean(), color='r', linestyle='--', label='Mean Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()