import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Configuration & Data Ingestion (Expanded Universe)
# ---------------------------------------------------------
TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
    'BHARTIARTL.NS', 'ITC.NS', 'WIPRO.NS', 'LT.NS', 'MARUTI.NS',
    'AXISBANK.NS', 'TITAN.NS'
]

START_DATE = '2020-01-01'
SPLIT_DATE = '2023-12-31'  # In-Sample (Train): 2020-2023, Out-of-Sample (Test): 2024-2026
END_DATE   = '2026-08-01'
RISK_FREE_RATE = 0.067
TRANSACTION_COST = 0.0015  # 15 bps friction per rebalance

print("Fetching NIFTY 50 market data from Yahoo Finance...")
raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE)

# Handle both 'Close' and 'Adj Close' column names safely
if 'Close' in raw_data.columns:
    price_data = raw_data['Close']
elif 'Adj Close' in raw_data.columns:
    price_data = raw_data['Adj Close']
else:
    price_data = raw_data

# Drop failed/empty columns automatically
data = price_data.dropna(axis=1, how='all').ffill().bfill()
TICKERS = list(data.columns)
num_assets = len(TICKERS)
print(f"[+] Active Universe ({num_assets} assets): {TICKERS}")

returns = np.log(data / data.shift(1)).dropna()

# Partition data into In-Sample (Train) and Out-of-Sample (Test)
train_returns = returns.loc[:SPLIT_DATE]
test_returns  = returns.loc[SPLIT_DATE:]

# ---------------------------------------------------------
# 2. Covariance Shrinkage (Pure NumPy)
# ---------------------------------------------------------
def shrinkage_covariance(returns_df, delta=0.20):
    sample_cov = returns_df.cov().values * 252
    prior_target = np.diag(np.diag(sample_cov))  # Diagonal target
    shrunk_cov = (1 - delta) * sample_cov + delta * prior_target
    return shrunk_cov

mu_train = train_returns.mean().values * 252
cov_train_shrunk = shrinkage_covariance(train_returns, delta=0.20)

# ---------------------------------------------------------
# 3. Optimization Routines
# ---------------------------------------------------------
def portfolio_performance(weights, mu, cov):
    ret = np.sum(weights * mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return ret, vol

def neg_sharpe_regularized(weights, mu, cov, rf=RISK_FREE_RATE, l2_gamma=0.10):
    ret, vol = portfolio_performance(weights, mu, cov)
    sr = (ret - rf) / vol
    l2_penalty = l2_gamma * np.sum(weights ** 2)
    return -(sr - l2_penalty)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
bounds = tuple((0.02, 0.20) for _ in range(num_assets))
init_weights = np.array([1.0 / num_assets] * num_assets)

# Execute SLSQP Optimization
opt_res = sco.minimize(
    neg_sharpe_regularized, 
    init_weights,
    args=(mu_train, cov_train_shrunk, RISK_FREE_RATE, 0.10),
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

w_optimal = opt_res.x

# ---------------------------------------------------------
# 4. Out-of-Sample Evaluation
# ---------------------------------------------------------
w_equal = np.array([1.0 / num_assets] * num_assets)

def evaluate_oos_performance(weights, test_ret_df, rf=RISK_FREE_RATE, cost=TRANSACTION_COST):
    daily_portfolio_ret = test_ret_df.dot(weights)
    ann_ret = (daily_portfolio_ret.mean() * 252) - cost
    ann_vol = daily_portfolio_ret.std() * np.sqrt(252)
    sharpe  = (ann_ret - rf) / ann_vol
    var_95 = -np.percentile(daily_portfolio_ret, 5)
    return ann_ret, ann_vol, sharpe, var_95

eq_ret, eq_vol, eq_sr, eq_var = evaluate_oos_performance(w_equal, test_returns)
opt_ret, opt_vol, opt_sr, opt_var = evaluate_oos_performance(w_optimal, test_returns)

# ---------------------------------------------------------
# 5. Output Results
# ---------------------------------------------------------
print("\n" + "="*65)
print("OUT-OF-SAMPLE EVALUATION RESULTS (TEST PERIOD: 2024 - 2026)")
print("="*65)
results_df = pd.DataFrame({
    'Metric': ['Annualized Net Return', 'Annualized Volatility', 'Out-of-Sample Sharpe Ratio', '1-Day 95% VaR'],
    'Equal-Weighted Baseline': [f"{eq_ret*100:.2f}%", f"{eq_vol*100:.2f}%", f"{eq_sr:.4f}", f"{eq_var*100:.2f}%"],
    'Regularized Shrunk Optimal': [f"{opt_ret*100:.2f}%", f"{opt_vol*100:.2f}%", f"{opt_sr:.4f}", f"{opt_var*100:.2f}%"]
})
print(results_df.to_string(index=False))

print("\n" + "="*65)
print("OPTIMAL ASSET WEIGHT ALLOCATION (V2)")
print("="*65)
weights_df = pd.DataFrame({'Ticker': TICKERS, 'Optimal Weight (%)': np.round(w_optimal * 100, 2)})
print(weights_df.sort_values(by='Optimal Weight (%)', ascending=False).to_string(index=False))

# Plot Out-of-Sample Performance
plt.figure(figsize=(10, 5))
cum_eq = (1 + test_returns.dot(w_equal)).cumprod()
cum_opt = (1 + test_returns.dot(w_optimal)).cumprod()

plt.plot(cum_eq, label='Equal-Weighted Baseline', linestyle='--')
plt.plot(cum_opt, label='Regularized Shrunk Optimal Portfolio', linewidth=2)
plt.title('Out-of-Sample Cumulative Performance (2024 - 2026)')
plt.xlabel('Date')
plt.ylabel('Growth of ₹1 Investment')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('oos_performance_comparison.png', dpi=300)
print("\n[+] Chart saved as 'oos_performance_comparison.png'")
