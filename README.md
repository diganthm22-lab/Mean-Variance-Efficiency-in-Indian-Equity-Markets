# NIFTY 50 Dynamic Portfolio Optimization and Mean-Variance Efficiency

An empirical quantitative finance project evaluating Markowitz Mean-Variance Portfolio Optimization on blue-chip Indian equities from the NIFTY 50 index (2020–2026). 

This project was developed as part of an academic research paper investigating portfolio variance minimization, Sharpe Ratio maximization via Sequential Least Squares Programming (SLSQP), and Monte Carlo risk simulations.

---

## Project Overview

* **Objective:** Find the optimal risk-adjusted portfolio weights ($w_i$) for a selected basket of Indian equities under a non-short-selling constraint ($w_i \ge 0$).
* **Data Source:** Daily historical adjusted close prices fetched via Yahoo Finance (`yfinance`).
* **Selected Assets:**
  * `HDFCBANK.NS` (Banking / Financials)
  * `HINDUNILVR.NS` (FMCG / Consumer Staples)
  * `MARUTI.NS` (Automobile)
  * `RELIANCE.NS` (Energy / Conglomerate)
  * `TCS.NS` (Information Technology)
* **Risk-Free Rate Assumption:** 6.7% (Indian 10-Year Government Securities Yield).

---

## Summary of Key Empirical Findings

| Portfolio Metric | Equal Weight (Baseline) | Optimal Portfolio (Max Sharpe) |
| :--- | :---: | :---: |
| **HDFC Bank Weight** | 20.00% | **0.00%** |
| **Hindustan Unilever Weight** | 20.00% | **0.00%** |
| **Maruti Suzuki Weight** | 20.00% | **46.96%** |
| **Reliance Industries Weight** | 20.00% | **0.00%** |
| **TCS Weight** | 20.00% | **53.04%** |
| **Expected Annual Return ($E[R_p]$)** | 10.23% | **14.61%** |
| **Annualized Volatility ($\sigma_p$)** | 17.51% | **24.07%** |
| **Sharpe Ratio ($R_f = 6.7\%$)** | 0.2016 | **0.3286** |

---

