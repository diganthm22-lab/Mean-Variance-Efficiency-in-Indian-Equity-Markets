# Out-of-Sample Efficiency, Covariance Shrinkage, and Regularized Portfolio Optimization in Indian Equity Markets

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2608.XXXXX-b31b1b.svg)](https://arxiv.org/)

This repository contains the replication codebase, empirical pipeline, and manuscript source files for the paper: **"Out-of-Sample Efficiency, Covariance Shrinkage, and Regularized Portfolio Optimization in Indian Equity Markets"**.

---

## Abstract

Classical Markowitz Mean-Variance Optimization often fails in real-world trading due to parameter estimation noise, the "Error Maximizer" paradox, and severe sector over-concentration. 

This project implements an institutionally robust quantitative portfolio optimization architecture applied to **12 blue-chip constituents of the NIFTY 50 index** across six major sectors (2020–2026). To eliminate corner-solution allocations and overfitting, we integrate:
1. **Ledoit-Wolf Linear Covariance Shrinkage** to stabilize noisy covariance matrices.
2. **$L_2$ Ridge Regularization** within the Sequential Least Squares Programming (SLSQP) solver.
3. **Hard Weight Bounds** ($2\% \le w_i \le 20\%$) to enforce mandatory cross-sector diversification.
4. **Walk-Forward Out-of-Sample Backtesting** (Training: 2020–2023; Testing: 2024–2026).
5. **Transaction Friction Accounting** (15 bps cost penalty per rebalance).

---

## Key Findings & Empirical Performance

Evaluated on unseen out-of-sample daily return data (January 2024 – August 2026) net of transaction costs:

| Metric | Equal-Weighted Baseline | Regularized Shrunk Optimal |
| :--- | :---: | :---: |
| **Annualized Net Return** | 2.29% | **3.59%** (+130 bps) |
| **Annualized Volatility** | **13.68%** | 14.35% |
| **Out-of-Sample Sharpe Ratio ($R_f=6.7\%$)** | -0.3221 | **-0.2167** |
| **1-Day 95% Historical VaR** | **1.34%** | 1.38% |

---

## Portfolio Asset Universe ($N=12$)

Capital is distributed across leading Indian equities representing key economic sectors:

- **Industrials / Capital Goods:** `LT.NS` (Larsen & Toubro)
- **Consumer Durables:** `TITAN.NS` (Titan Company)
- **FMCG / Staples:** `ITC.NS` (ITC Limited)
- **Telecommunications:** `BHARTIARTL.NS` (Bharti Airtel)
- **Information Technology:** `INFY.NS` (Infosys), `TCS.NS`, `WIPRO.NS`
- **Financial Services:** `AXISBANK.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`
- **Automobiles:** `MARUTI.NS` (Maruti Suzuki)
- **Energy / Conglomerates:** `RELIANCE.NS` (Reliance Industries)

---

## Mathematical Framework

### 1. Linear Covariance Shrinkage
The sample covariance matrix $\mathbf{\Sigma}_{\text{sample}}$ is shrunk toward a diagonal target matrix $\mathbf{F} = \text{diag}(\mathbf{\Sigma}_{\text{sample}})$:
$$\mathbf{\Sigma}_{\text{shrunk}} = (1 - \delta) \mathbf{\Sigma}_{\text{sample}} + \delta \mathbf{F}, \quad \delta = 0.20$$

### 2. Regularized Objective Function
$$\min_{\mathbf{w}} \quad -\left( \frac{\mathbf{w}^T \mathbf{\mu}_{\text{train}} - R_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma}_{\text{shrunk}} \mathbf{w}}} \right) + \lambda \sum_{i=1}^N w_i^2$$
Subject to:
$$\sum_{i=1}^N w_i = 1.0, \quad 0.02 \le w_i \le 0.20 \quad \forall i$$

