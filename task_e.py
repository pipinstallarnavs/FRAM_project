import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm

# --- 1. Load Frozen Market State ---
try:
    state_df = pd.read_csv("market_state.csv")
    S0 = float(state_df['S0'].iloc[0])
    sigma = float(state_df['Sigma'].iloc[0])
    r = float(state_df['RiskFreeRate'].iloc[0])
    ticker = str(state_df['Ticker'].iloc[0])
    print(f"Loaded Market State: S0={S0:.2f}, Vol={sigma:.2%}")
    
    df_pricing = pd.read_csv("task_b_pricing.csv")
except FileNotFoundError:
    print("Error: Run Task_B.py first.")
    exit()

# --- 2. Get Historical Data (Last 60 Days) ---
print(f"Fetching historical returns for {ticker}...")
# We need enough data to get 60 returns
history = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)

if isinstance(history.columns, pd.MultiIndex):
    history = history['Close'].iloc[:, 0] if 'Close' in history else history.iloc[:, 0]
elif 'Close' in history.columns:
     history = history['Close']
else:
    history = history.iloc[:, 0]

history = history.squeeze().dropna()
# Calculate Log Returns
daily_returns = np.log(history / history.shift(1)).dropna()
last_60_returns = daily_returns.tail(60).values

print(f"Loaded {len(last_60_returns)} days of historical returns.")

# --- 3. Define Portfolio (Short Straddle) ---
# Unhedged: Short 100 ATM Calls, Short 100 ATM Puts
qty = -100
target_days = 30
call_row = df_pricing[(df_pricing['Maturity (Days)'] == target_days) & (df_pricing['Strike Label'] == "ATM")].iloc[0]
put_row  = df_pricing[(df_pricing['Maturity (Days)'] == target_days) & (df_pricing['Strike Label'] == "ATM")].iloc[0]

K = call_row['Strike Price']
T_years = target_days / 365.0

# --- 4. BSM Helper Function ---
def bsm_price(S, K, T, r, sigma, kind):
    # Safety for non-positive S in simulation
    if S <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == 'call': return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def get_delta(S, K, T, r, sigma, kind):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if kind == 'call': return norm.cdf(d1)
    else: return norm.cdf(d1) - 1

# --- 5. Calculate Current Greeks ---
# Greeks for Unhedged Portfolio
c_delta = get_delta(S0, K, T_years, r, sigma, 'call')
p_delta = get_delta(S0, K, T_years, r, sigma, 'put')
port_delta = (qty * c_delta) + (qty * p_delta)

# Hedge Requirement (Delta Neutral)
hedge_shares = -port_delta

print(f"\n--- Portfolio Setup ---")
print(f"Unhedged Delta: {port_delta:.2f}")
print(f"Hedge: Buy/Sell {hedge_shares:.2f} shares")

# --- 6. METHOD A: Parametric VaR (Delta-Normal) ---
# Formula: VaR = |Position_Value| * Vol * Z_score
# Ideally: VaR = |Portfolio_Delta * S0| * Daily_Vol * Z_score

daily_vol = sigma / np.sqrt(252)
z_95 = 1.645
z_99 = 2.326

# Unhedged Parametric VaR
exposure_unhedged = abs(port_delta * S0)
var_param_95_unhedged = exposure_unhedged * daily_vol * z_95
var_param_99_unhedged = exposure_unhedged * daily_vol * z_99

# Hedged Parametric VaR
# Theoretically, Delta-Hedged Parametric VaR is 0 (first order). 
# Remaining risk is Gamma, which Parametric VaR ignores so we should get 0. If not we can re run later.
var_param_95_hedged = 0.0
var_param_99_hedged = 0.0

# --- 7. METHOD B: Historical Simulation (Full Revaluation) ---
unhedged_pnls = []
hedged_pnls = []

current_val_unhedged = (qty * call_row['Call Price']) + (qty * put_row['Put Price'])

for ret in last_60_returns:
    # Simulate Stock Price Move
    S_sim = S0 * np.exp(ret)
    
    # Re-price Options
    sim_call = bsm_price(S_sim, K, T_years, r, sigma, 'call')
    sim_put  = bsm_price(S_sim, K, T_years, r, sigma, 'put')
    
    # PnL Calculation
    pnl_opt = (qty * (sim_call - call_row['Call Price'])) + (qty * (sim_put - put_row['Put Price']))
    pnl_stock = hedge_shares * (S_sim - S0)
    
    unhedged_pnls.append(pnl_opt)
    hedged_pnls.append(pnl_opt + pnl_stock)

# Calculate Percentiles (5th for 95%, 1st for 99%)
var_hist_95_unhedged = -np.percentile(unhedged_pnls, 5)
var_hist_99_unhedged = -np.percentile(unhedged_pnls, 1)

var_hist_95_hedged = -np.percentile(hedged_pnls, 5)
var_hist_99_hedged = -np.percentile(hedged_pnls, 1)

# --- 8. Deliverable Table ---
results = pd.DataFrame({
    'Metric': ['Parametric VaR (95%)', 'Parametric VaR (99%)', 
               'Historical VaR (95%)', 'Historical VaR (99%)'],
    'Unhedged Portfolio': [var_param_95_unhedged, var_param_99_unhedged, 
                           var_hist_95_unhedged, var_hist_99_unhedged],
    'Hedged Portfolio': [var_param_95_hedged, var_param_99_hedged, 
                         var_hist_95_hedged, var_hist_99_hedged]
})

print("\n--- Part E: Value-at-Risk (VaR) Analysis ---")
print(results.round(2))

results.to_csv("task_e_var.csv", index=False)
print("\nVaR results saved to 'task_e_var.csv'")