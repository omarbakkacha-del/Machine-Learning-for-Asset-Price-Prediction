# Machine Learning for Asset Price Prediction

A machine learning project for predicting financial asset returns using historical market data, financial feature engineering, classical baseline models, XGBoost, and LSTM neural networks.

The project follows a complete pipeline:

**Raw Data → Feature Engineering → Prediction → Evaluation → Walk-Forward Validation → Backtesting**

---

## Objectives

The main objective is to investigate whether machine learning models can improve the prediction of future financial asset returns compared with simple baseline methods.

The project focuses on:

* Historical market data
* Financial feature engineering
* Return prediction
* Machine learning
* Deep learning
* Out-of-sample evaluation
* Walk-forward validation
* Trading strategy backtesting

---

## Project Pipeline

```text
Market Data
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Baseline Models
     │
     ▼
XGBoost
     │
     ▼
LSTM
     │
     ▼
Model Evaluation
     │
     ▼
Walk-Forward Validation
     │
     ▼
Backtesting
     │
     ▼
Final Model Comparison
```

---

# Phases

## Phase 1 — Data Collection & Cleaning

Collect historical OHLCV data for selected financial assets and prepare a clean time-series dataset.

Main variables:

* Open
* High
* Low
* Close
* Adjusted Close
* Volume

---

## Phase 2 — Feature Engineering

Transform raw market data into variables that can be used by the prediction models.

Features include:

* Log returns
* Lagged returns
* Moving averages
* Rolling volatility
* Momentum
* RSI
* MACD
* Volume features

The primary prediction target is the future return:

$$
r_{t+1} =\log\left(\frac{P_{t+1}}{P_t}\right)
$$

where \(P_t\) is the asset price at time \(t\).

---

## Phase 4 — XGBoost

Train an XGBoost regression model using the engineered financial features.

The model learns a function of the form:

$$
\hat r_{t+1}=f_\theta(X_t)
$$

where \(X_t\) contains information available at time \(t\).

---

## Phase 5 — LSTM

Implement a Long Short-Term Memory neural network to learn temporal dependencies from sequences of historical observations.

The model receives a sequence:

$$
(X_{t-L+1},\ldots,X_t)
$$

and predicts the future return:

$$
\hat r_{t+1}
$$

---

## Phase 6 — Model Evaluation

Evaluate every model using the same unseen test data.

Metrics include:

* MAE
* RMSE
* \(R^2\)
* Directional Accuracy

All models are evaluated under the same conditions to ensure a meaningful comparison.

---

## Phase 7 — Walk-Forward Validation

Because financial data is time-dependent, random train/test splitting is avoided.

Instead, models are repeatedly trained on past data and evaluated on subsequent unseen observations.

```text
TRAIN ───────── TEST

TRAIN ─────────────── TEST

TRAIN ─────────────────── TEST

TRAIN ──────────────────────── TEST
```

This helps prevent look-ahead bias and provides a more realistic estimate of out-of-sample performance.

---

## Phase 8 — Backtesting

Convert model predictions into trading signals and evaluate whether the predictions can produce a useful trading strategy.

Performance metrics include:

* Cumulative Return
* Annualized Return
* Sharpe Ratio
* Maximum Drawdown
* Win Rate
* Turnover
* Transaction Costs

A simple long/short signal can be defined using the predicted return:

$$
w_t =
\begin{cases}
1 & \text{if } \hat r_{t+1}>0,\\
-1 & \text{if } \hat r_{t+1}\leq0.
\end{cases}
$$

Transaction costs are incorporated into the backtest.

---

## Phase 9 — Final Comparison

All models are compared using both predictive and financial performance.

| Model             | MAE | RMSE | Directional Accuracy | Sharpe Ratio | Max Drawdown |
| ----------------- | --: | ---: | -------------------: | -----------: | -----------: |
| Random Walk       |     |      |                      |              |              |
| Linear Regression |     |      |                      |              |              |
| XGBoost           |     |      |                      |              |              |
| LSTM              |     |      |                      |              |              |

The final objective is to determine whether more sophisticated models provide meaningful improvements over the baseline approaches.

---

# Data

Raw datasets are stored in:

```text
data/raw/
```

Raw data should remain unchanged.

Cleaned and transformed datasets are stored in:

```text
data/processed/
```

Example raw dataset:

```text
Date,Open,High,Low,Close,Volume
2020-01-02,74.06,75.15,73.80,75.09,135480400
...
```

---

# Project Structure

```text
machine-learning-asset-price-prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_collection_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_lstm.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_walk_forward_validation.ipynb
│   ├── 08_backtesting.ipynb
│   └── 09_final_comparison.ipynb
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── backtesting.py
│
├── models/
│   └── saved_models/
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
│
└── tests/
    └── test_pipeline.py
```

---

# Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* PyTorch
* Matplotlib

---

# Experimental Methodology

Every model follows the same general procedure:

```text
Data
  ↓
Training
  ↓
Validation
  ↓
Testing
  ↓
Evaluation
  ↓
Comparison
```

