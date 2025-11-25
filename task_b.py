import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# --- 1. Fetch Data & Freeze Parameters ---
ticker = "TATAMOTORS.NS"
print(f"Fetching data for {ticker}...")
stock_data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)

# Clean Data
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data = stock_data['Close'].iloc[:, 0] if 'Close' in stock_data else stock_data.iloc[:, 0]
elif 'Close' in stock_data.columns:
     stock_data = stock_data['Close']
else:
    stock_data = stock_data.iloc[:, 0]
stock_data = stock_data.squeeze().dropna()

# Capture "Frozen" Parameters
S0 = float(stock_data.iloc[-1])
log_returns = np.log(stock_data / stock_data.shift(1)).dropna()
sigma = float(log_returns.std() * np.sqrt(252))
r = 0.07 

market_state = pd.DataFrame([{
    'S0': S0,
    'Sigma': sigma,
    'RiskFreeRate': r,
    'Ticker': ticker
}])
market_state.to_csv("market_state.csv", index=False)
print(f"Market State Frozen: S0={S0:.2f}, Vol={sigma:.2%}")

# --- 2. BSM Function ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- 3. Generate Option Chain ---
strikes_pct = [-0.05, -0.02, 0.00, 0.02, 0.05]
strike_labels = ["ATM-5%", "ATM-2%", "ATM", "ATM+2%", "ATM+5%"]
strikes = [S0 * (1 + pct) for pct in strikes_pct]
days_list = [30, 60, 90]

results = []
for days in days_list:
    T = days / 365.0
    for K, label in zip(strikes, strike_labels):
        call_price = black_scholes(S0, K, T, r, sigma, 'call')
        put_price = black_scholes(S0, K, T, r, sigma, 'put')
        results.append({
            "Maturity (Days)": days,
            "Strike Label": label,
            "Strike Price": round(K, 2),
            "Call Price": round(call_price, 2),
            "Put Price": round(put_price, 2)
        })

pricing_df = pd.DataFrame(results)
pricing_df.to_csv("task_b_pricing.csv", index=False)
print("Option Prices saved to 'task_b_pricing.csv'")