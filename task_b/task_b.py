import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# -----------------------------
# Config / Parameters
# -----------------------------
ticker = "TATAMOTORS.NS"
TRADING_DAYS = 252        # use trading days consistently
r = 0.07                  # annual risk-free rate (continuous compounding)
period_str = "3mo"

# --- 1. Fetch Data & Extract Close ---
print(f"Fetching data for {ticker}...")
stock_df = yf.download(ticker, period=period_str, interval="1d", auto_adjust=True, progress=False)

# Robust extraction of Close / single column:
if isinstance(stock_df.columns, pd.MultiIndex):
    # Try to extract level 'Close' explicitly
    try:
        close_df = stock_df.xs('Close', axis=1, level=0, drop_level=False)
        # if multiple tickers, take the first second-level column
        stock_series = close_df.iloc[:, 0]
    except Exception:
        # fallback to first column overall
        stock_series = stock_df.iloc[:, 0]
else:
    if 'Close' in stock_df.columns:
        stock_series = stock_df['Close']
    else:
        stock_series = stock_df.iloc[:, 0]

stock_series = stock_series.squeeze().dropna()

# Basic sanity checks
if stock_series.size == 0:
    raise ValueError("No price data downloaded for ticker - check ticker or internet connection.")
if stock_series.size < 5:
    print("Warning: very little historical data available.")

# --- Freeze parameters ---
S0 = float(stock_series.iloc[-1])
log_returns = np.log(stock_series / stock_series.shift(1)).dropna()
sigma = float(log_returns.std(ddof=0) * np.sqrt(TRADING_DAYS))   # ddof=0 for population std (choice)
market_state = pd.DataFrame([{
    'S0': S0,
    'Sigma': sigma,
    'RiskFreeRate': r,
    'Ticker': ticker
}])
market_state.to_csv("market_state.csv", index=False)
print(f"Market State Frozen: S0={S0:.2f}, Vol={sigma:.2%}")

# --- BSM (vectorized) ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Vectorized: S, K, T, sigma can be scalars or numpy arrays of same shape.
    Returns price(s).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Handle T == 0 (or negative)
    price = np.zeros_like(S, dtype=float)

    # intrinsic where T <= 0
    mask_zero = (T <= 0)
    if np.any(mask_zero):
        if option_type == 'call':
            price[mask_zero] = np.maximum(0.0, S[mask_zero] - K[mask_zero])
        else:
            price[mask_zero] = np.maximum(0.0, K[mask_zero] - S[mask_zero])

    # normal BSM where T > 0
    mask = ~mask_zero
    if np.any(mask):
        sqrtT = np.sqrt(T[mask])
        d1 = (np.log(S[mask] / K[mask]) + (r + 0.5 * sigma[mask]**2) * T[mask]) / (sigma[mask] * sqrtT)
        d2 = d1 - sigma[mask] * sqrtT
        if option_type == 'call':
            price[mask] = S[mask] * norm.cdf(d1) - K[mask] * np.exp(-r * T[mask]) * norm.cdf(d2)
        else:
            price[mask] = K[mask] * np.exp(-r * T[mask]) * norm.cdf(-d2) - S[mask] * norm.cdf(-d1)

    # If user passed scalars, return scalar
    if price.shape == ():
        return float(price)
    return price

# --- Generate Option Chain ---
strikes_pct = [-0.05, -0.02, 0.00, 0.02, 0.05]
strike_labels = ["ATM-5%", "ATM-2%", "ATM", "ATM+2%", "ATM+5%"]
strikes = [S0 * (1 + pct) for pct in strikes_pct]
days_list = [30, 60, 90]

results = []
for days in days_list:
    T = days / TRADING_DAYS   # consistent with sigma annualization
    for K, label in zip(strikes, strike_labels):
        call_price = black_scholes(S0, K, T, r, sigma, 'call')
        put_price  = black_scholes(S0, K, T, r, sigma, 'put')
        results.append({
            "Maturity (Days)": days,
            "Strike Label": label,
            "Strike Price": round(K, 2),
            "Call Price": round(float(call_price), 2),
            "Put Price": round(float(put_price), 2)
        })

pricing_df = pd.DataFrame(results)
pricing_df.to_csv("task_b_pricing.csv", index=False)
print("Option Prices saved to 'task_b_pricing.csv'")
