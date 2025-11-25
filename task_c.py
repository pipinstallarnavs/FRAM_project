import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf # Only used for Task 8 (Market IV comparison), not for Greek Calc

# --- 1. Load Frozen State ---
try:
    state_df = pd.read_csv("market_state.csv")
    S0 = float(state_df['S0'].iloc[0])
    sigma_hist = float(state_df['Sigma'].iloc[0])
    r = float(state_df['RiskFreeRate'].iloc[0])
    ticker = str(state_df['Ticker'].iloc[0])
    print(f"Loaded Market State: S0={S0:.2f}, Vol={sigma_hist:.2%}")
    
    df_pricing = pd.read_csv("task_b_pricing.csv")
    print("Loaded Pricing Table.")
except FileNotFoundError:
    print("Error: Run Task_B.py first to generate data files.")
    exit()

# --- 2. Calculate Greeks (Using Frozen S0 and Sigma) ---
def calculate_greeks(row, S, r, sigma):
    K = row['Strike Price']
    T = row['Maturity (Days)'] / 365.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    # Gamma & Vega are same for Call/Put
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100 

    # We calculate Greeks for the CALL option as standard deliverable
    # (Assignment asks for "Delta (call & put)", here we show Call Delta)
    delta_call = norm.cdf(d1)
    delta_put  = norm.cdf(d1) - 1
    
    theta_call = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    
    return pd.Series([delta_call, delta_put, gamma, vega, theta_call, rho_call])

print("Calculating Greeks...")
greeks_cols = ['Delta_Call', 'Delta_Put', 'Gamma', 'Vega', 'Theta', 'Rho']
df_pricing[greeks_cols] = df_pricing.apply(
    lambda row: calculate_greeks(row, S0, r, sigma_hist), axis=1
)

print("\n--- Part C: Greeks Table (Snippet) ---")
print(df_pricing[['Maturity (Days)', 'Strike Label', 'Delta_Call', 'Delta_Put', 'Gamma']].head())

df_pricing.to_csv("task_c_greeks.csv", index=False)
print("Greeks saved to 'task_c_greeks.csv'")

# --- 3. Market IV & Vol Surface (Separate Logic) ---
# This part MUST fetch new data because we are comparing our model to the Real World.
print(f"\nFetching real market data for IV comparison...")
try:
    tk = yf.Ticker(ticker)
    exps = tk.options
    if exps:
        opt_chain = tk.option_chain(exps[0])
        calls = opt_chain.calls
        
        # Plot Volatility Smile
        plt.figure(figsize=(10, 6))
        plt.plot(calls['strike'], calls['impliedVolatility'], 'o-', label=f'Exp: {exps[0]}')
        plt.title(f"Volatility Smile: {ticker} (Real Market Data)")
        plt.xlabel("Strike Price")
        plt.ylabel("Implied Volatility (IV)")
        plt.grid(True)
        plt.legend()
        plt.savefig("volatility_smile.png")
        print("Saved 'volatility_smile.png'")
    else:
        print("No option chain data available via API.")
except Exception as e:
    print(f"Market Data Fetch Failed: {e}")