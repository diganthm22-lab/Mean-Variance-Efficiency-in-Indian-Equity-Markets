import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA ACQUISITION & SAFE PREPARATION
# ==========================================
# 20 High-Liquidity NIFTY 50 Blue Chip Constituents
NIFTY_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "HCLTECH.NS", "LT.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "NTPC.NS", "POWERGRID.NS"
]

def fetch_data(tickers, start_date="2019-01-01", end_date="2026-08-01"):
    print("Fetching historical price data from Yahoo Finance...")
    price_data = {}
    
    # Fetch ticker by ticker to prevent cross-asset NaN propagation
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Handle YFinance multi-index or single-index Close/Adj Close
                if 'Adj Close' in df.columns:
                    series = df['Adj Close']
                elif 'Close' in df.columns:
                    series = df['Close']
                else:
                    series = df.iloc[:, 0]
                
                # Convert to 1D Series if returned as DataFrame
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                    
                price_data[ticker] = series
        except Exception as e:
            print(f"Warning: Could not fetch {ticker}: {e}")

    # Combine into a single DataFrame
    prices = pd.DataFrame(price_data)
    
    # Forward fill missing values (e.g. holidays), then backfill
    prices = prices.ffill().bfill().dropna(axis=1)
    
    # Compute Log Daily Returns
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Data successfully loaded: {returns.shape[1]} assets across {returns.shape[0]} trading days.")
    return returns

# ==========================================
# 2. CORE OPTIMIZATION ENGINE
# ==========================================
def optimize_portfolio(R_train, Rf=0.067, l2_reg=0.02, w_bounds=(0.0, 0.15)):
    N = R_train.shape[1]
    mu_train = R_train.mean().values * 252
    
    # Analytical Ledoit-Wolf Covariance Shrinkage
    lw = LedoitWolf()
    lw.fit(R_train.values)
    Sigma_shrunk = lw.covariance_ * 252  # Annualized
    
    def objective(w):
        port_ret = np.dot(w, mu_train)
        port_vol = np.sqrt(np.maximum(1e-8, np.dot(w.T, np.dot(Sigma_shrunk, w))))
        sharpe = (port_ret - Rf) / port_vol
        l2_penalty = l2_reg * np.sum(w**2)
        return -sharpe + l2_penalty

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [w_bounds for _ in range(N)]
    init_w = np.array([1.0 / N] * N)

    res = minimize(
        objective, 
        init_w, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    return res.x if res.success else init_w

# ==========================================
# 3. DYNAMIC ROLLING-WINDOW BACKTEST ENGINE
# ==========================================
def run_rolling_backtest(returns, lookback_months=36, fee_bps=15):
    fee_pct = fee_bps / 10000.0
    dates = returns.index
    N = returns.shape[1]
    
    # Resample to month-start dates
    monthly_dates = returns.resample('MS').first().index
    
    portfolio_returns = []
    equal_weight_returns = []
    
    current_w = np.array([1.0 / N] * N)
    ew_w = np.array([1.0 / N] * N)
    
    print("Running dynamic rolling-window optimization backtest...")
    
    for i in range(len(monthly_dates) - 1):
        t_start = monthly_dates[i]
        t_end = monthly_dates[i+1]
        
        train_start = t_start - pd.DateOffset(months=lookback_months)
        
        # Skip until full 36-month lookback is available
        if train_start < dates[0]:
            continue
            
        R_train = returns.loc[train_start:t_start]
        R_test = returns.loc[t_start:t_end].iloc[:-1]

        if R_train.empty or R_test.empty:
            continue

        # Re-optimize portfolio weights
        new_w = optimize_portfolio(R_train, Rf=0.067, l2_reg=0.02, w_bounds=(0.0, 0.15))
        
        # Calculate Turnover & Transaction Friction
        turnover = np.sum(np.abs(new_w - current_w))
        friction_cost = turnover * fee_pct
        
        # Portfolio Returns
        opt_rets = R_test.dot(new_w).copy()
        ew_rets = R_test.dot(ew_w).copy()
        
        # Deduct friction on rebalance day
        if len(opt_rets) > 0:
            opt_rets.iloc[0] -= friction_cost
            
        portfolio_returns.append(opt_rets)
        equal_weight_returns.append(ew_rets)
        current_w = new_w

    if not portfolio_returns:
        raise ValueError("No backtest results generated. Check input date ranges.")

    oos_opt = pd.concat(portfolio_returns)
    oos_ew = pd.concat(equal_weight_returns)
    return oos_opt, oos_ew

# ==========================================
# 4. METRIC COMPUTATION & DISPLAY
# ==========================================
def compute_metrics(returns_series, Rf=0.067):
    ann_ret = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = (ann_ret - Rf) / ann_vol
    var_95 = np.percentile(returns_series, 5) * -1
    
    # Cumulative Maximum Drawdown
    cum_rets = np.exp(returns_series.cumsum())
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        "Annualized Net Return": f"{ann_ret * 100:.2f}%",
        "Annualized Volatility": f"{ann_vol * 100:.2f}%",
        "Sharpe Ratio (Rf=6.7%)": f"{sharpe:.4f}",
        "1-Day 95% VaR": f"{var_95 * 100:.2f}%",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%"
    }

# ==========================================
# 5. MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    returns = fetch_data(NIFTY_TICKERS)
    opt_rets, ew_rets = run_rolling_backtest(returns)
    
    opt_metrics = compute_metrics(opt_rets)
    ew_metrics = compute_metrics(ew_rets)
    
    results_df = pd.DataFrame({
        "Equal-Weighted Baseline (1/N)": ew_metrics,
        "Regularized Shrunk Optimal": opt_metrics
    })
    
    print("\n" + "="*55)
    print("        OUT-OF-SAMPLE BACKTEST RESULTS (2022-2026)")
    print("="*55)
    print(results_df)
    print("="*55)
    
    # Generate and save equity curve
    cum_opt = np.exp(opt_rets.cumsum())
    cum_ew = np.exp(ew_rets.cumsum())
    
    plt.figure(figsize=(10, 5))
    plt.plot(cum_ew, label="Equal-Weighted Baseline (1/N)", linestyle="--", color="gray")
    plt.plot(cum_opt, label="Regularized Shrunk Optimal Portfolio", color="#1f77b4", linewidth=2)
    plt.title("Out-of-Sample Dynamic Cumulative Performance (2022 - 2026)")
    plt.xlabel("Date")
    plt.ylabel("Growth of ₹1 Investment")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("oos_performance.png", dpi=300)
    print("\nChart saved successfully to 'oos_performance.png'.")
    
