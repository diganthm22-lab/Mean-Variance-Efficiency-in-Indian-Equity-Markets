# Dynamic Rolling Out-of-Sample Portfolio Optimization in Indian Equity Markets

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete open-source Python implementation and LaTeX manuscript for the paper **"Dynamic Rolling Out-of-Sample Portfolio Optimization, Covariance Shrinkage, and $L_2$ Regularization in Indian Equity Markets"**.

## Core Quantitative Features
- **Dataset**: 20 High-Liquidity Blue-Chip Constituents from the NIFTY 50 Index (2019–2026).
- **Analytical Ledoit-Wolf Covariance Shrinkage**: Replaces noisy sample covariance matrices with mathematically optimal target shrinkage estimators ($\Sigma_{\text{shrunk}}$).
- **$L_2$ Ridge Penalty & Box Constraints**: Prevents extreme asset concentration and corner solutions ($0\% \le w_i \le 15\%$).
- **Dynamic Rolling-Window Rebalancing**: 36-month sliding lookback window with monthly re-optimization.
- **Turnover & Transaction Friction**: Realized execution cost deduction ($15\text{ bps} \times \text{Turnover}$).

## Empirical Performance Summary (2022–2026 Backtest)

| Metric | Naive Equal-Weighted ($1/N$) | Regularized Shrunk Optimal |
| :--- | :---: | :---: |
| **Annualized Net Return** | **9.14%** | 8.95% |
| **Annualized Volatility** | **12.81%** | 14.11% |
| **Sharpe Ratio ($R_f=6.7\%$)** | **0.1902** | 0.1598 |
| **1-Day 95% Historical VaR** | **1.26%** | 1.36% |
| **Maximum Drawdown** | **-15.90%** | -18.12% |

## Getting Started

### Prerequisites
Install the required dependencies inside your virtual environment:

```bash
pip install numpy pandas yfinance scikit-learn scipy matplotlib
