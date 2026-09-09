from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"

TICKER = "AAPL"

N_BOOTSTRAPS = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

TRADING_DAYS_PER_YEAR = 252

# Block length in trading days.
BLOCK_LENGTH = 10


# ============================================================
# DATA
# ============================================================

def load_backtest(model_name: str) -> pd.DataFrame:
    """
    Load Phase 7 out-of-sample backtest results.
    """

    file_path = (
        PREDICTIONS_DIR
        / f"{TICKER}_{model_name}_backtest.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Backtest file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# MOVING-BLOCK BOOTSTRAP
# ============================================================

def moving_block_bootstrap(
    returns: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one moving-block bootstrap sample.

    Blocks are contiguous observations of length
    `block_length`.

    The resulting sample has the same length as
    the original time series.
    """

    n = len(returns)

    if block_length <= 0:
        raise ValueError(
            "Block length must be positive."
        )

    if block_length > n:
        raise ValueError(
            "Block length cannot exceed "
            "number of observations."
        )

    # Number of possible overlapping blocks.
    n_blocks = n - block_length + 1

    # Number of blocks required.
    blocks_needed = int(
        np.ceil(n / block_length)
    )

    sampled_starts = rng.integers(
        0,
        n_blocks,
        size=blocks_needed
    )

    bootstrap_sample = []

    for start in sampled_starts:

        block = returns[
            start:start + block_length
        ]

        bootstrap_sample.append(block)

    bootstrap_sample = np.concatenate(
        bootstrap_sample
    )

    return bootstrap_sample[:n]


# ============================================================
# STATISTICS
# ============================================================

def calculate_sharpe(
    returns: np.ndarray
) -> float:
    """
    Annualized Sharpe ratio.
    """

    std = returns.std(ddof=1)

    if std == 0:
        return np.nan

    return (
        returns.mean()
        / std
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def calculate_cumulative_return(
    returns: np.ndarray
) -> float:
    """
    Convert cumulative log return to simple return.
    """

    return np.exp(
        returns.sum()
    ) - 1


# ============================================================
# BOOTSTRAP CI FOR A STATISTIC
# ============================================================

def bootstrap_statistic(
    returns: pd.Series,
    statistic_function,
    block_length: int,
) -> dict:
    """
    Moving-block bootstrap confidence interval
    for an arbitrary statistic.
    """

    returns = (
        returns
        .dropna()
        .to_numpy()
    )

    observed = statistic_function(
        returns
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_values = np.empty(
        N_BOOTSTRAPS
    )

    for i in range(N_BOOTSTRAPS):

        sample = moving_block_bootstrap(
            returns,
            block_length,
            rng
        )

        bootstrap_values[i] = (
            statistic_function(sample)
        )

    alpha = (
        1 - CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_values,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_values,
        1 - alpha / 2
    )

    return {
        "observed": observed,
        "ci_lower": lower,
        "ci_upper": upper,
    }


# ============================================================
# MODEL ANALYSIS
# ============================================================

def analyze_model(
    model_name: str
) -> dict:

    print()
    print("=" * 60)
    print(
        f"DEPENDENCE-AWARE ANALYSIS — "
        f"{model_name.upper()}"
    )
    print("=" * 60)

    df = load_backtest(
        model_name
    )

    returns = df[
        "strategy_return"
    ]

    mean_result = bootstrap_statistic(
        returns,
        np.mean,
        BLOCK_LENGTH
    )

    sharpe_result = bootstrap_statistic(
        returns,
        calculate_sharpe,
        BLOCK_LENGTH
    )

    cumulative_result = bootstrap_statistic(
        returns,
        calculate_cumulative_return,
        BLOCK_LENGTH
    )

    print("\nMean Strategy Return")
    print("-" * 60)

    print(
        f"Observed: "
        f"{mean_result['observed']:.8f}"
    )

    print(
        f"95% MBB CI: "
        f"[{mean_result['ci_lower']:.8f}, "
        f"{mean_result['ci_upper']:.8f}]"
    )

    print("\nSharpe Ratio")
    print("-" * 60)

    print(
        f"Observed: "
        f"{sharpe_result['observed']:.6f}"
    )

    print(
        f"95% MBB CI: "
        f"[{sharpe_result['ci_lower']:.6f}, "
        f"{sharpe_result['ci_upper']:.6f}]"
    )

    print("\nCumulative Return")
    print("-" * 60)

    print(
        f"Observed: "
        f"{cumulative_result['observed']:.4%}"
    )

    print(
        f"95% MBB CI: "
        f"[{cumulative_result['ci_lower']:.4%}, "
        f"{cumulative_result['ci_upper']:.4%}]"
    )

    return {
        "model": model_name,
        "mean_daily_return":
            mean_result["observed"],
        "mean_return_ci_lower":
            mean_result["ci_lower"],
        "mean_return_ci_upper":
            mean_result["ci_upper"],
        "sharpe_ratio":
            sharpe_result["observed"],
        "sharpe_ci_lower":
            sharpe_result["ci_lower"],
        "sharpe_ci_upper":
            sharpe_result["ci_upper"],
        "cumulative_return":
            cumulative_result["observed"],
        "cumulative_return_ci_lower":
            cumulative_result["ci_lower"],
        "cumulative_return_ci_upper":
            cumulative_result["ci_upper"],
        "observations":
            len(df),
    }


# ============================================================
# DIRECT STRATEGY COMPARISON
# ============================================================

def compare_strategies(
    returns_a: pd.Series,
    returns_b: pd.Series,
    name_a: str,
    name_b: str,
) -> dict:
    """
    Dependence-aware comparison between two strategies.

    We bootstrap the paired difference:

        D_t = R_t^A - R_t^B

    Null hypothesis:

        E[D_t] = 0
    """

    combined = pd.concat(
        [
            returns_a.rename("a"),
            returns_b.rename("b"),
        ],
        axis=1
    ).dropna()

    differences = (
        combined["a"]
        - combined["b"]
    ).to_numpy()

    observed_difference = (
        differences.mean()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_means = np.empty(
        N_BOOTSTRAPS
    )

    for i in range(N_BOOTSTRAPS):

        sample = moving_block_bootstrap(
            differences,
            BLOCK_LENGTH,
            rng
        )

        bootstrap_means[i] = (
            sample.mean()
        )

    alpha = (
        1 - CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_means,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2
    )

    print()
    print("=" * 60)
    print(
        f"STRATEGY COMPARISON — "
        f"{name_a.upper()} vs {name_b.upper()}"
    )
    print("=" * 60)

    print(
        f"\nObserved mean return difference: "
        f"{observed_difference:.8f}"
    )

    print(
        f"95% MBB CI: "
        f"[{lower:.8f}, {upper:.8f}]"
    )

    if lower > 0:

        print(
            "Conclusion: evidence that "
            f"{name_a} outperforms {name_b}."
        )

    elif upper < 0:

        print(
            "Conclusion: evidence that "
            f"{name_a} underperforms {name_b}."
        )

    else:

        print(
            "Conclusion: no statistically "
            "clear difference."
        )

    return {
        "comparison":
            f"{name_a}_vs_{name_b}",
        "mean_return_difference":
            observed_difference,
        "ci_lower":
            lower,
        "ci_upper":
            upper,
        "observations":
            len(differences),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "PHASE 7.2 — "
        "DEPENDENCE-AWARE STATISTICAL INFERENCE"
    )
    print("=" * 60)

    print(
        f"\nBootstrap samples: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"Block length: "
        f"{BLOCK_LENGTH} trading days"
    )

    # --------------------------------------------------------
    # Load strategies
    # --------------------------------------------------------

    xgb = load_backtest(
        "xgboost"
    )

    lstm = load_backtest(
        "lstm"
    )

    # Buy-and-hold return is reconstructed
    # from the actual market return.

    buy_hold_returns = (
        xgb["actual_return"]
    )

    # --------------------------------------------------------
    # Individual strategy inference
    # --------------------------------------------------------

    xgb_results = analyze_model(
        "xgboost"
    )

    lstm_results = analyze_model(
        "lstm"
    )

    buy_hold_results = analyze_model_from_series(
        "buy_and_hold",
        buy_hold_returns
    )

    # --------------------------------------------------------
    # Pairwise comparisons
    # --------------------------------------------------------

    xgb_vs_lstm = compare_strategies(
        xgb["strategy_return"],
        lstm["strategy_return"],
        "xgboost",
        "lstm"
    )

    xgb_vs_buy_hold = compare_strategies(
        xgb["strategy_return"],
        buy_hold_returns,
        "xgboost",
        "buy_and_hold"
    )

    lstm_vs_buy_hold = compare_strategies(
        lstm["strategy_return"],
        buy_hold_returns,
        "lstm",
        "buy_and_hold"
    )

    # --------------------------------------------------------
    # Save individual results
    # --------------------------------------------------------

    individual_results = pd.DataFrame([
        xgb_results,
        lstm_results,
        buy_hold_results,
    ])

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    individual_path = (
        METRICS_DIR
        / f"{TICKER}_dependence_aware_metrics.csv"
    )

    individual_results.to_csv(
        individual_path,
        index=False
    )

    # --------------------------------------------------------
    # Save pairwise results
    # --------------------------------------------------------

    comparison_results = pd.DataFrame([
        xgb_vs_lstm,
        xgb_vs_buy_hold,
        lstm_vs_buy_hold,
    ])

    comparison_path = (
        METRICS_DIR
        / f"{TICKER}_strategy_comparisons.csv"
    )

    comparison_results.to_csv(
        comparison_path,
        index=False
    )

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        f"\nIndividual results:\n"
        f"{individual_path}"
    )

    print(
        f"\nStrategy comparisons:\n"
        f"{comparison_path}"
    )


def analyze_model_from_series(
    model_name: str,
    returns: pd.Series
) -> dict:
    """
    Same dependence-aware analysis as analyze_model(),
    but accepts an arbitrary return series.
    """

    print()
    print("=" * 60)
    print(
        f"DEPENDENCE-AWARE ANALYSIS — "
        f"{model_name.upper()}"
    )
    print("=" * 60)

    mean_result = bootstrap_statistic(
        returns,
        np.mean,
        BLOCK_LENGTH
    )

    sharpe_result = bootstrap_statistic(
        returns,
        calculate_sharpe,
        BLOCK_LENGTH
    )

    cumulative_result = bootstrap_statistic(
        returns,
        calculate_cumulative_return,
        BLOCK_LENGTH
    )

    print("\nMean Strategy Return")
    print("-" * 60)

    print(
        f"Observed: "
        f"{mean_result['observed']:.8f}"
    )

    print(
        f"95% MBB CI: "
        f"[{mean_result['ci_lower']:.8f}, "
        f"{mean_result['ci_upper']:.8f}]"
    )

    print("\nSharpe Ratio")
    print("-" * 60)

    print(
        f"Observed: "
        f"{sharpe_result['observed']:.6f}"
    )

    print(
        f"95% MBB CI: "
        f"[{sharpe_result['ci_lower']:.6f}, "
        f"{sharpe_result['ci_upper']:.6f}]"
    )

    print("\nCumulative Return")
    print("-" * 60)

    print(
        f"Observed: "
        f"{cumulative_result['observed']:.4%}"
    )

    print(
        f"95% MBB CI: "
        f"[{cumulative_result['ci_lower']:.4%}, "
        f"{cumulative_result['ci_upper']:.4%}]"
    )

    return {
        "model": model_name,
        "mean_daily_return":
            mean_result["observed"],
        "mean_return_ci_lower":
            mean_result["ci_lower"],
        "mean_return_ci_upper":
            mean_result["ci_upper"],
        "sharpe_ratio":
            sharpe_result["observed"],
        "sharpe_ci_lower":
            sharpe_result["ci_lower"],
        "sharpe_ci_upper":
            sharpe_result["ci_upper"],
        "cumulative_return":
            cumulative_result["observed"],
        "cumulative_return_ci_lower":
            cumulative_result["ci_lower"],
        "cumulative_return_ci_upper":
            cumulative_result["ci_upper"],
        "observations":
            len(returns),
    }


if __name__ == "__main__":
    main()