import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ['HDFCBANK.NS', 'RELIANCE.NS', 'TCS.NS', 'MARUTI.NS', 'HINDUNILVR.NS']
start_date = '2020-01-01'
end_date = '2026-01-01'

raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
data = raw_data['Adj Close']

# 1. Daily Log Returns & Annualized Metrics
log_returns = np.log(data / data.shift(1)).dropna()
annual_returns = log_returns.mean() * 252
cov_matrix = log_returns.cov() * 252
risk_free_rate = 0.067  # 6.7% Indian 10-Year G-Sec Yield

def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.sum(weights * returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

# 2. Monte Carlo Simulation
num_portfolios = 15000
results = np.zeros((3, num_portfolios))
weights_record = []

np.random.seed(42) # For reproducible results
for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights) 
    weights_record.append(weights)
    
    p_return, p_volatility = portfolio_performance(weights, annual_returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_volatility
    
    results[0, i] = p_return
    results[1, i] = p_volatility
    results[2, i] = sharpe_ratio

# 3. Optimization: Find Maximum Sharpe Ratio Portfolio
def neg_sharpe_ratio(weights, returns, cov_matrix, rf):
    p_ret, p_vol = portfolio_performance(weights, returns, cov_matrix)
    return -(p_ret - rf) / p_vol

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) 
bounds = tuple((0, 1) for _ in range(len(tickers))) # No short-selling (weights between 0 and 1)
initial_guess = len(tickers) * [1.0 / len(tickers)] # Equal weight starting point

opt_sharpe = minimize(neg_sharpe_ratio, initial_guess, 
                      args=(annual_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

max_sharpe_weights = opt_sharpe.x
opt_ret, opt_vol = portfolio_performance(max_sharpe_weights, annual_returns, cov_matrix)
max_sharpe_ratio = (opt_ret - risk_free_rate) / opt_vol

print("\n==========================================")
print("     OPTIMAL PORTFOLIO ALLOCATION         ")
print("==========================================")
for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"{ticker:<15}: {weight*100:.2f}%")

print(f"\nExpected Annual Return : {opt_ret*100:.2f}%")
print(f"Annualized Volatility  : {opt_vol*100:.2f}%")
print(f"Maximum Sharpe Ratio   : {max_sharpe_ratio:.4f}")

#4. Plot
plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(opt_vol, opt_ret, color='black', marker='x', s=200, label='Optimal Portfolio (Max Sharpe)')
plt.title('Efficient Frontier: NIFTY 50 Portfolio')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Expected Return')
plt.legend()
plt.grid(True)
plt.show()
