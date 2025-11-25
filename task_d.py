import pandas as pd
import numpy as np
from scipy.stats import norm

# --- 1. Load Frozen Market State ---
try:
    state_df = pd.read_csv("market_state.csv")
    S0 = float(state_df['S0'].iloc[0])
    sigma = float(state_df['Sigma'].iloc[0])
    r = float(state_df['RiskFreeRate'].iloc[0])
    print(f"Loaded Market State: S0={S0:.2f}, Vol={sigma:.2%}")
    
    df_pricing = pd.read_csv("task_b_pricing.csv")
except FileNotFoundError:
    print("Error: Run Task_B.py first.")
    exit()

# --- 2. Greek Calculation  ---
def get_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return delta, gamma, vega

# --- 3. Construct Initial Portfolio  ---

# Scenario: Short Straddle (Short 30-day ATM Call & Put)
target_days = 30
target_label = "ATM"
qty_initial = -100 

# Get Option Data
opt_call = df_pricing[(df_pricing['Maturity (Days)'] == target_days) & (df_pricing['Strike Label'] == target_label)].iloc[0]
opt_put  = df_pricing[(df_pricing['Maturity (Days)'] == target_days) & (df_pricing['Strike Label'] == target_label)].iloc[0]
K = opt_call['Strike Price']
T_years = target_days / 365.0

# Calculate Initial Greeks
c_delta, c_gamma, c_vega = get_greeks(S0, K, T_years, r, sigma, 'call')
p_delta, p_gamma, p_vega = get_greeks(S0, K, T_years, r, sigma, 'put')

# Initial Portfolio Greeks
initial_delta = (qty_initial * c_delta) + (qty_initial * p_delta)
initial_gamma = (qty_initial * c_gamma) + (qty_initial * p_gamma)
initial_vega  = (qty_initial * c_vega) + (qty_initial * p_vega)

print(f"\n--- 1. Initial Portfolio (Short Straddle) ---")
print(f"Delta: {initial_delta:.2f} | Gamma: {initial_gamma:.4f} | Vega: {initial_vega:.2f}")

# --- 4. Step A: Gamma Hedge (Using Options) ---

# We are 'Short Gamma' (negative). ie, we need to buy options.

# 90-day ATM Call is hedging instrument.
hedge_days = 90
hedge_opt = df_pricing[(df_pricing['Maturity (Days)'] == hedge_days) & (df_pricing['Strike Label'] == "ATM")].iloc[0]
K_hedge = hedge_opt['Strike Price']
T_hedge = hedge_days / 365.0

# Get Greeks of the Hedge Instrument
h_delta, h_gamma, h_vega = get_greeks(S0, K_hedge, T_hedge, r, sigma, 'call')

# Calculate Quantity needed to neutralize Gamma
qty_gamma_hedge = -initial_gamma / h_gamma

print(f"\n--- 2. Gamma Hedge Execution ---")
print(f"Instrument: Long 90-Day ATM Call")
print(f"Required Qty: {qty_gamma_hedge:.2f}")

# --- 5. Step B: Delta Hedge (Using Stock) ---

# Net_Delta = Initial_Delta + (Qty_Gamma_Hedge * Hedge_Delta)
intermediate_delta = initial_delta + (qty_gamma_hedge * h_delta)

# We hedge the remaining Delta with Stock (Delta = 1)
qty_stock_hedge = -intermediate_delta

print(f"\n--- 3. Delta Hedge Execution ---")
print(f"New Net Delta (after Gamma hedge): {intermediate_delta:.2f}")
print(f"Action: Trade {qty_stock_hedge:.2f} shares of Underlying Stock")

# --- 6. Final Comparison Table (Before vs After) ---
final_delta = intermediate_delta + (qty_stock_hedge * 1.0) 
final_gamma = initial_gamma + (qty_gamma_hedge * h_gamma)  
final_vega  = initial_vega + (qty_gamma_hedge * h_vega)    

comparison = pd.DataFrame({
    'Metric': ['Delta', 'Gamma', 'Vega'],
    'Before Hedge': [initial_delta, initial_gamma, initial_vega],
    'After Gamma Hedge': [intermediate_delta, 0.0, final_vega], 
    'After Delta Hedge (Final)': [final_delta, final_gamma, final_vega] 
})

print("\n--- 4. Greeks Comparison: Before vs After ---")
print(comparison.round(4))

# --- 7. Full PnL Simulation (Backtesting the Double Hedge) ---
shocks = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
results = []

def bsm_price(S, K, T, r, sigma, kind):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == 'call': return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

print("\n--- 5. PnL Simulation (Double Hedged) ---")
for shock in shocks:
    S_new = S0 * (1 + shock)
    
    # 1. Original Portfolio PnL
    new_call = bsm_price(S_new, K, T_years, r, sigma, 'call')
    new_put  = bsm_price(S_new, K, T_years, r, sigma, 'put')
    pnl_orig = qty_initial * (new_call - opt_call['Call Price']) + qty_initial * (new_put - opt_put['Put Price'])
    
    # 2. Gamma Hedge PnL (The 90-day Option)
    new_hedge_opt = bsm_price(S_new, K_hedge, T_hedge, r, sigma, 'call')
    pnl_gamma_hedge = qty_gamma_hedge * (new_hedge_opt - hedge_opt['Call Price'])
    
    # 3. Delta Hedge PnL (The Stock)
    pnl_stock_hedge = qty_stock_hedge * (S_new - S0)
    
    # Total
    total_pnl = pnl_orig + pnl_gamma_hedge + pnl_stock_hedge
    
    results.append({
        "Move": f"{shock*100:+.0f}%",
        "Unhedged PnL": round(pnl_orig, 2),
        "Total Hedged PnL": round(total_pnl, 2),
        "Improvement": round(total_pnl - pnl_orig, 2)
    })

sim_df = pd.DataFrame(results)
print(sim_df)
sim_df.to_csv("task_d_full_hedge.csv", index=False)