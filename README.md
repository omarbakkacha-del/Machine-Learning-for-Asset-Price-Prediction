# Machine Learning for Asset Price Prediction

A reproducible machine-learning and deep-learning study of next-day financial return prediction using historical market data, engineered technical features, XGBoost, and LSTM neural networks.

The project evaluates not only predictive accuracy but also whether model predictions translate into economically meaningful **out-of-sample trading performance**.

---

## Research Question

> **Can machine learning and deep learning models predict future financial asset returns effectively out of sample, and do their predictions translate into meaningful trading performance after transaction costs?**

The project focuses on a central difficulty of financial machine learning:

$$
\text{good in-sample prediction}
\;\not\Rightarrow\;
\text{good out-of-sample trading performance}.
$$

The methodology therefore emphasizes chronological validation, walk-forward testing, transaction costs, and statistical inference.

---

# Pipeline

```text
Historical Market Data
        │
        ▼
   Data Cleaning
        │
        ▼
 Feature Engineering
        │
        ├───────────────┐
        ▼               ▼
    XGBoost           LSTM
        │               │
        └───────┬───────┘
                ▼
        Out-of-Sample
        Model Evaluation
                │
                ▼
       Walk-Forward Validation
                │
                ▼
          Backtesting
                │
                ▼
      Statistical Testing
                │
                ▼
 Dependence-Aware Inference
                │
                ▼
       Final Comparison
```

---

# Objectives

The project investigates:

* whether engineered market features contain predictive information about next-day returns;
* whether tree-based models outperform recurrent neural networks;
* whether predictive metrics translate into profitable trading signals;
* how model performance changes under repeated out-of-sample testing;
* whether observed trading performance is statistically distinguishable from noise;
* whether temporal dependence affects statistical conclusions.

The study deliberately avoids claiming predictive success merely from favorable in-sample or single-period results.

---

# Dataset

The current experiment uses historical **Apple Inc. (AAPL)** market data obtained from Yahoo Finance through `yfinance`.

The raw dataset contains daily:

* Open
* High
* Low
* Close
* Adjusted Close
* Volume

The raw data covers approximately 2015–2025.

Raw data is stored in:

```text
data/raw/AAPL.csv
```

Cleaned data is stored in:

```text
data/processed/AAPL_clean.csv
```

Feature-engineered data is stored in:

```text
data/processed/AAPL_features.csv
```

---

# Mathematical Formulation

Let \(P_t\) denote the closing price at time \(t\).

The prediction target is the next-day logarithmic return

$$r_{t+1}=
\log\left(\frac{P_{t+1}}{P_t}\right).
$$

At time \(t\), the model observes a feature vector

$$
X_t\in\mathbb{R}^{28}.
$$

The learning problem is therefore

$$
f_\theta(X_t)\approx r_{t+1}.
$$

For the LSTM, the model instead receives a temporal sequence

$$
(X_{t-L+1},\ldots,X_t),
$$

with sequence length

$$
L=20.
$$

The objective is to estimate the conditional behavior of

$$
r_{t+1}\mid X_t
$$

without using information from the future.

---

# Phase 1 — Data Collection and Cleaning

Historical AAPL OHLCV data is downloaded using `yfinance`.

The cleaning pipeline:

1. removes the Yahoo Finance header artifacts;
2. converts dates to datetime objects;
3. converts numerical variables to numeric types;
4. removes invalid observations;
5. removes missing price observations;
6. removes non-positive prices;
7. sorts observations chronologically;
8. removes duplicate dates.

The cleaned dataset contains 2,766 observations.

Implementation:

```text
src/data.py
```

---

# Phase 2 — Feature Engineering

The feature-engineering pipeline constructs 28 predictive variables.

### Return features

* log return
* 1-day lagged return
* 2-day lagged return
* 3-day lagged return
* 5-day lagged return
* 10-day lagged return

### Moving-average features

* 5-day moving average
* 10-day moving average
* 20-day moving average
* 50-day moving average

and normalized price-to-moving-average ratios.

### Volatility

Rolling volatility is calculated over:

* 5 days
* 20 days
* 50 days

### Momentum

Momentum features are calculated over:

* 5 days
* 10 days
* 20 days

### Technical indicators

* RSI
* MACD
* MACD signal
* MACD histogram

### Volume features

* percentage volume change
* 20-day volume moving average
* relative volume

### Intraday features

* daily high-low range
* intraday return

The resulting feature dataset contains 2,715 observations and 36 columns, including the original market variables and target.

Implementation:

```text
src/features.py
```

---

# Phase 3 — XGBoost

The first predictive model is an XGBoost regression model.

The model estimates

$$
\hat r_{t+1}=f_\theta(X_t).
$$

The implemented configuration is:

```text
n_estimators = 300
max_depth = 4
learning_rate = 0.03
subsample = 0.8
colsample_bytree = 0.8
objective = reg:squarederror
random_state = 42
```

Implementation:

```text
src/XGBoost.py
```

The trained model is saved as:

```text
models/saved_models/AAPL_xgboost.json
```

---

# Phase 4 — LSTM

The second model is a Long Short-Term Memory neural network.

Rather than observing a single feature vector, the LSTM receives a sequence of the previous 20 observations:

$$
(X_{t-19},\ldots,X_t).
$$

Architecture:

```text
Input size:       28
Sequence length:  20
Hidden size:      64
LSTM layers:      2
Dropout:          0.2
Output:           1
```

Training configuration:

```text
Batch size:       32
Learning rate:    0.001
Epochs:           30
Random seed:      42
```

Feature standardization is performed using a scaler fitted only on the training data.

Implementation:

```text
src/lstm.py
```

The trained model is saved as:

```text
models/saved_models/AAPL_lstm.pt
```

---

# Phase 5 — Model Evaluation

Predictive performance is evaluated using:

### Mean Absolute Error

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat y_i|.
$$

### Root Mean Squared Error

$$
RMSE=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
}.
$$

### Coefficient of determination

$$
R^2=
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}.
$$

### Directional Accuracy

$$
DA=
\frac{1}{n}
\sum_i
\mathbf{1}
\left[
\operatorname{sign}(\hat y_i)=
\operatorname{sign}(y_i)
\right].
$$

Implementation:

```text
src/evaluation.py
```

---

# Phase 6 — Walk-Forward Validation

Financial observations are temporally dependent, so ordinary random train/test splitting is inappropriate for the main experiment.

The project therefore uses **expanding-window walk-forward validation**.

The initial training window contains 60% of the available observations.

Each subsequent test window contains 60 trading observations.

Conceptually:

```text
Fold 1

TRAIN
|----------------------------------|
                                  | TEST |
                                  | 60   |


Fold 2

TRAIN
|----------------------------------------|
                                        | TEST |
                                        | 60   |


Fold 3

TRAIN
|----------------------------------------------|
                                              | TEST |
                                              | 60   |
```

The training window expands after every fold.

This produces:

```text
Number of folds:       19
Total OOS observations: 1086
```

Both XGBoost and LSTM are evaluated on exactly the same chronological out-of-sample observations.

The LSTM's first test sequence is constructed using observations immediately preceding the test period, which are part of the historical training information and therefore do not introduce look-ahead bias.

Implementation:

```text
src/walk_forward.py
```

---

# Phase 7 — Backtesting

Predictions are converted into a simple long/short trading signal:

$$w_t=
\operatorname{sign}(\hat r_{t+1}).$$

Transaction costs are incorporated according to

$$
R^{strategy}_{t+1}
=
w_t r_{t+1}
-
c|w_t-w_{t-1}|,
$$

where

$$
c=0.001.
$$

Thus the backtest does not assume costless trading.

Portfolio wealth is computed from cumulative log returns:

$$
W_t
=
\exp
\left(
\sum_{i\leq t}
R_i^{strategy}
\right).
$$

Performance metrics include:

* cumulative return;
* annualized return;
* annualized volatility;
* Sharpe ratio;
* maximum drawdown;
* win rate;
* number of position changes.

Implementation:

```text
src/backtesting.py
```

---

# Phase 7.1 — Statistical Backtest Validation

Observed trading performance can arise from sampling variation.

The project therefore performs statistical tests in addition to reporting point estimates.

Directional accuracy is tested using an exact binomial test:

$$
H_0:p=0.5.
$$

Bootstrap confidence intervals are also calculated for:

* mean daily strategy return;
* Sharpe ratio;
* cumulative return.

10,000 bootstrap samples are used with a fixed random seed.

Implementation:

```text
src/statistical_testing.py
```

---

# Phase 7.2 — Dependence-Aware Statistical Inference

Financial returns are not necessarily independent across time.

To account for temporal dependence, the project additionally uses a **moving-block bootstrap**.

The current experiment uses:

$$
L=10
$$

trading days per block and 10,000 bootstrap samples.

Confidence intervals are calculated for:

* mean strategy return;
* Sharpe ratio;
* cumulative return.

Paired comparisons are also performed between:

* XGBoost and LSTM;
* XGBoost and buy-and-hold;
* LSTM and buy-and-hold.

Implementation:

```text
src/dependence_aware_testing.py
```

> **Note:** block-length sensitivity analysis was not performed in this version of the experiment. The value \(L=10\) should therefore be regarded as a methodological choice rather than a fully sensitivity-validated parameter.

---

# Phase 8 — Final Model Comparison

The final comparison combines predictive and economic performance.

## Out-of-Sample Results

| Model   |      MAE |     RMSE |   \(R^2\) | Directional Accuracy |
| ------- | -------: | -------: | --------: | -------------------: |
| XGBoost | 0.013213 | 0.018095 | -0.047490 |               48.99% |
| LSTM    | 0.014573 | 0.019856 | -0.261253 |               47.33% |

XGBoost outperforms LSTM on all four predictive metrics.

However, neither model achieves directional accuracy significantly above 50%.

---

## Trading Results

| Strategy   | Cumulative Return | Annualized Return | Volatility | Sharpe | Max Drawdown |
| ---------- | ----------------: | ----------------: | ---------: | -----: | -----------: |
| XGBoost    |           +19.24% |            +4.17% |     28.11% |  0.145 |      -42.93% |
| LSTM       |           -68.54% |           -23.54% |     28.13% | -0.954 |      -76.25% |
| Buy & Hold |           +76.93% |           +14.16% |     28.08% |  0.472 |      -33.43% |

The results show that XGBoost produces a positive cumulative return, but substantially underperforms the buy-and-hold benchmark.

The LSTM strategy performs substantially worse, generating a large negative cumulative return.

---

# Statistical Conclusions

The statistical analysis provides a more cautious interpretation.

For XGBoost:

* directional accuracy: 48.99%;
* exact binomial test \(p=0.524\);
* no evidence of directional accuracy different from 50%;
* moving-block-bootstrap CI for mean return includes zero;
* moving-block-bootstrap CI for Sharpe includes zero.

Therefore, the observed positive XGBoost return does **not establish a statistically reliable trading edge**.

For LSTM:

* directional accuracy: 47.33%;
* exact binomial test \(p=0.084\);
* moving-block-bootstrap CI for mean return is negative;
* moving-block-bootstrap CI for Sharpe is negative;
* moving-block-bootstrap CI for cumulative return is negative.

The dependence-aware analysis therefore provides evidence that the LSTM strategy performs poorly.

For pairwise comparisons:

* XGBoost vs LSTM: the observed advantage of XGBoost is not statistically conclusive under the moving-block bootstrap;
* XGBoost vs buy-and-hold: the observed underperformance is not statistically conclusive under the tested mean-return comparison;
* LSTM vs buy-and-hold: there is evidence of underperformance.

---

# Main Findings

The experiment leads to four principal conclusions.

### 1. XGBoost outperforms LSTM

XGBoost achieves lower MAE and RMSE, a less negative \(R^2\), and higher directional accuracy.

### 2. Predictive superiority does not imply trading superiority

Although XGBoost is the better predictive model in this experiment, its trading strategy achieves only:

$$
Sharpe=0.145,
$$

compared with

$$
Sharpe=0.472
$$

for buy-and-hold.

### 3. Neither model demonstrates a robust predictive edge

The XGBoost directional accuracy is approximately 49%, below the 50% reference level.

Its negative \(R^2\) also indicates that its squared prediction error is worse than the constant-mean benchmark over the evaluated out-of-sample sample.

### 4. The LSTM performs particularly poorly

The LSTM generates:

$$
-68.54\%
$$

cumulative return and a Sharpe ratio of

$$
-0.954.
$$

The dependence-aware inference provides evidence that this poor performance is not simply explained by ordinary sampling variation under the implemented procedure.

---

# Interpretation

The experiment does **not** support the hypothesis that the implemented XGBoost or LSTM models provide a robust exploitable forecasting advantage for AAPL next-day returns.

The more defensible conclusion is:

> **For this dataset, feature set, model configuration, validation procedure, and transaction-cost assumption, XGBoost substantially outperforms LSTM but neither model demonstrates statistically convincing predictive or economic superiority over the simple buy-and-hold benchmark.**

This negative result is an important part of the experiment rather than a failure of the methodology.

Financial return prediction is a particularly difficult machine-learning problem because small apparent forecasting signals can disappear under strict out-of-sample evaluation and trading costs.

---

# Limitations

The current experiment has several important limitations.

### Single asset

The experiment currently focuses on AAPL.

Results should not be generalized to other equities, asset classes, or market regimes without additional experiments.

### Limited feature universe

The models use historical price, return, volatility, momentum, technical-indicator, and volume features.

They do not currently incorporate:

* macroeconomic variables;
* fundamentals;
* options-implied volatility;
* order-book information;
* news;
* sentiment;
* cross-sectional information.

### Hyperparameter selection

The current hyperparameters are fixed configurations rather than the result of a large nested hyperparameter optimization procedure.

### Transaction-cost model

The backtest uses a simplified transaction-cost assumption:

$$
c=0.001.
$$

Real transaction costs can depend on:

* bid-ask spreads;
* market impact;
* liquidity;
* order size;
* broker fees;
* market conditions.

### Statistical inference

The moving-block bootstrap uses a block length of 10 trading days.

Sensitivity to alternative block lengths has not yet been evaluated.

### Multiple testing

The experiment does not implement a comprehensive correction for multiple model specifications, feature sets, or strategy configurations.

Consequently, future experimentation should be careful to distinguish exploratory discoveries from confirmatory evidence.

---

# Reproducibility

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main pipeline can be executed phase by phase.

### Data

```bash
python src/data.py
```

### Features

```bash
python src/features.py
```

### XGBoost

```bash
python src/XGBoost.py
```

### LSTM

```bash
python src/lstm.py
```

### Walk-forward validation

```bash
python src/walk_forward.py
```

### Backtesting

```bash
python src/backtesting.py
```

### Statistical testing

```bash
python src/statistical_testing.py
```

### Dependence-aware inference

```bash
python src/dependence_aware_testing.py
```

### Final comparison

```bash
python src/final_comparison.py
```

The resulting metrics and figures are stored under:

```text
results/
```

---

# Repository Structure

```text
Machine-Learning-for-Asset-Price-Prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── AAPL.csv
│   └── processed/
│       ├── AAPL_clean.csv
│       └── AAPL_features.csv
│
├── models/
│   └── saved_models/
│       ├── AAPL_xgboost.json
│       └── AAPL_lstm.pt
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── XGBoost.py
│   ├── lstm.py
│   ├── evaluation.py
│   ├── walk_forward.py
│   ├── backtesting.py
│   ├── statistical_testing.py
│   ├── dependence_aware_testing.py
│   └── final_comparison.py
│
└── results/
    ├── metrics/
    ├── predictions/
    ├── walk_forward/
    └── figures/
```

---

# Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* PyTorch
* SciPy
* Matplotlib
* yfinance

---

# Future Research

Possible extensions include:

* testing multiple assets;
* cross-sectional equity prediction;
* transformer-based sequence models;
* temporal convolutional networks;
* richer market and macroeconomic features;
* volatility forecasting;
* probabilistic prediction;
* calibrated uncertainty estimates;
* regime-dependent models;
* nested walk-forward hyperparameter optimization;
* alternative transaction-cost models;
* block-length sensitivity analysis;
* statistical tests designed specifically for comparing predictive forecasts;
* multiple-testing corrections;
* portfolio construction across multiple assets.

---

# Disclaimer

This project is intended for **research and educational purposes only**.

The results do not constitute financial advice, investment advice, or a recommendation to buy or sell any financial asset.

Past simulated performance does not imply future performance.

No claim is made that the models developed in this repository can reliably predict future financial markets or generate profitable trading strategies.
