from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"

TICKER = "AAPL"

N_BOOTSTRAPS = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

TRADING_DAYS_PER_YEAR = 252


def load_backtest(model_name: str) -> pd.DataFrame:
    """
    Load the out-of-sample backtest generated during Phase 7.
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


def directional_accuracy_test(
    df: pd.DataFrame
) -> dict:
    """
    Exact binomial test for directional accuracy.

    H0: p = 0.5
    H1: p != 0.5
    """

    actual_sign = np.sign(
        df["actual_return"]
    )

    predicted_sign = np.sign(
        df["predicted_return"]
    )

    correct = (
        actual_sign == predicted_sign
    )

    valid = (
        predicted_sign != 0
    )

    correct = correct[valid]

    n = len(correct)

    successes = int(correct.sum())

    directional_accuracy = (
        successes / n
    )

    test = binomtest(
        successes,
        n,
        p=0.5,
        alternative="two-sided"
    )

    confidence_interval = test.proportion_ci(
        confidence_level=CONFIDENCE_LEVEL
    )

    return {
        "n": n,
        "correct_predictions": successes,
        "directional_accuracy":
            directional_accuracy,
        "null_probability": 0.5,
        "p_value": test.pvalue,
        "ci_lower":
            confidence_interval.low,
        "ci_upper":
            confidence_interval.high,
    }


def mean_return_test(
    returns: pd.Series
) -> dict:
    """
    Bootstrap confidence interval for mean
    strategy return.
    """

    returns = (
        returns
        .dropna()
        .to_numpy()
    )

    n = len(returns)

    observed_mean = returns.mean()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_means = np.empty(
        N_BOOTSTRAPS
    )

    for i in range(N_BOOTSTRAPS):

        sample = rng.choice(
            returns,
            size=n,
            replace=True
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

    return {
        "mean_daily_return":
            observed_mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "n": n,
    }


def calculate_sharpe(
    returns: np.ndarray
) -> float:

    std = returns.std(ddof=1)

    if std == 0:
        return np.nan

    return (
        returns.mean()
        / std
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def bootstrap_sharpe(
    returns: pd.Series
) -> dict:

    returns = (
        returns
        .dropna()
        .to_numpy()
    )

    observed_sharpe = calculate_sharpe(
        returns
    )

    n = len(returns)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_sharpes = np.empty(
        N_BOOTSTRAPS
    )

    for i in range(N_BOOTSTRAPS):

        sample = rng.choice(
            returns,
            size=n,
            replace=True
        )

        bootstrap_sharpes[i] = (
            calculate_sharpe(sample)
        )

    alpha = (
        1 - CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_sharpes,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_sharpes,
        1 - alpha / 2
    )

    return {
        "sharpe_ratio":
            observed_sharpe,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def bootstrap_cumulative_return(
    returns: pd.Series
) -> dict:

    returns = (
        returns
        .dropna()
        .to_numpy()
    )

    n = len(returns)

    observed_return = (
        np.exp(returns.sum()) - 1
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_returns = np.empty(
        N_BOOTSTRAPS
    )

    for i in range(N_BOOTSTRAPS):

        sample = rng.choice(
            returns,
            size=n,
            replace=True
        )

        bootstrap_returns[i] = (
            np.exp(sample.sum()) - 1
        )

    alpha = (
        1 - CONFIDENCE_LEVEL
    )

    lower = np.quantile(
        bootstrap_returns,
        alpha / 2
    )

    upper = np.quantile(
        bootstrap_returns,
        1 - alpha / 2
    )

    return {
        "cumulative_return":
            observed_return,
        "ci_lower": lower,
        "ci_upper": upper,
    }


def analyze_model(
    model_name: str
) -> dict:

    print()
    print("=" * 60)
    print(
        f"STATISTICAL ANALYSIS — "
        f"{model_name.upper()}"
    )
    print("=" * 60)

    df = load_backtest(
        model_name
    )

    # --------------------------------------------------------
    # Directional accuracy
    # --------------------------------------------------------

    direction = directional_accuracy_test(
        df
    )

    print("\nDirectional Accuracy")
    print("-" * 60)

    print(
        f"Accuracy: "
        f"{direction['directional_accuracy']:.4%}"
    )

    print(
        f"95% CI: "
        f"[{direction['ci_lower']:.4%}, "
        f"{direction['ci_upper']:.4%}]"
    )

    print(
        f"Binomial p-value: "
        f"{direction['p_value']:.6f}"
    )

    if direction["p_value"] < 0.05:
        print(
            "Result: statistically different "
            "from 50% at the 5% level."
        )
    else:
        print(
            "Result: not statistically different "
            "from 50% at the 5% level."
        )

    # --------------------------------------------------------
    # Mean return
    # --------------------------------------------------------

    mean_test = mean_return_test(
        df["strategy_return"]
    )

    print("\nMean Strategy Return")
    print("-" * 60)

    print(
        f"Mean daily return: "
        f"{mean_test['mean_daily_return']:.8f}"
    )

    print(
        f"95% bootstrap CI: "
        f"[{mean_test['ci_lower']:.8f}, "
        f"{mean_test['ci_upper']:.8f}]"
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    sharpe = bootstrap_sharpe(
        df["strategy_return"]
    )

    print("\nSharpe Ratio")
    print("-" * 60)

    print(
        f"Observed Sharpe: "
        f"{sharpe['sharpe_ratio']:.6f}"
    )

    print(
        f"95% bootstrap CI: "
        f"[{sharpe['ci_lower']:.6f}, "
        f"{sharpe['ci_upper']:.6f}]"
    )

    # --------------------------------------------------------
    # Cumulative return
    # --------------------------------------------------------

    cumulative = bootstrap_cumulative_return(
        df["strategy_return"]
    )

    print("\nCumulative Return")
    print("-" * 60)

    print(
        f"Observed cumulative return: "
        f"{cumulative['cumulative_return']:.4%}"
    )

    print(
        f"95% bootstrap CI: "
        f"[{cumulative['ci_lower']:.4%}, "
        f"{cumulative['ci_upper']:.4%}]"
    )

    return {
        "model": model_name,

        "directional_accuracy":
            direction["directional_accuracy"],

        "directional_accuracy_ci_lower":
            direction["ci_lower"],

        "directional_accuracy_ci_upper":
            direction["ci_upper"],

        "directional_accuracy_p_value":
            direction["p_value"],

        "mean_daily_return":
            mean_test["mean_daily_return"],

        "mean_return_ci_lower":
            mean_test["ci_lower"],

        "mean_return_ci_upper":
            mean_test["ci_upper"],

        "sharpe_ratio":
            sharpe["sharpe_ratio"],

        "sharpe_ci_lower":
            sharpe["ci_lower"],

        "sharpe_ci_upper":
            sharpe["ci_upper"],

        "cumulative_return":
            cumulative["cumulative_return"],

        "cumulative_return_ci_lower":
            cumulative["ci_lower"],

        "cumulative_return_ci_upper":
            cumulative["ci_upper"],

        "observations":
            len(df),
    }


def main():

    print("=" * 60)
    print(
        "PHASE 7.1 — "
        "STATISTICAL BACKTEST VALIDATION"
    )
    print("=" * 60)

    xgb_results = analyze_model(
        "xgboost"
    )

    lstm_results = analyze_model(
        "lstm"
    )

    results = pd.DataFrame([
        xgb_results,
        lstm_results,
    ])

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        METRICS_DIR
        / f"{TICKER}_statistical_tests.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        results[
            [
                "model",
                "directional_accuracy",
                "directional_accuracy_p_value",
                "mean_daily_return",
                "sharpe_ratio",
                "cumulative_return",
            ]
        ].to_string(index=False)
    )

    print()

    print(
        f"Statistical results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()