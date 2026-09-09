from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"

TICKER = "AAPL"


# ============================================================
# LOAD DATA
# ============================================================

def load_walk_forward_predictions(
    model_name: str
) -> pd.DataFrame:

    file_path = (
        WALK_FORWARD_DIR
        / f"{TICKER}_{model_name}_predictions.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Walk-forward predictions not found: "
            f"{file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_backtest(
    model_name: str
) -> pd.DataFrame:

    file_path = (
        PREDICTIONS_DIR
        / f"{TICKER}_{model_name}_backtest.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Backtest file not found: "
            f"{file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_walk_forward_metrics(
    model_name: str
) -> pd.DataFrame:

    file_path = (
        WALK_FORWARD_DIR
        / f"{TICKER}_{model_name}_metrics.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Walk-forward metrics not found: "
            f"{file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# PREDICTION METRICS
# ============================================================

def calculate_prediction_metrics(
    df: pd.DataFrame
) -> dict:

    actual = df["actual_return"].to_numpy()
    predicted = df["predicted_return"].to_numpy()

    errors = actual - predicted

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    denominator = np.sum(
        (actual - actual.mean()) ** 2
    )

    if denominator == 0:
        r2 = np.nan
    else:
        r2 = (
            1
            - np.sum(errors ** 2)
            / denominator
        )

    actual_sign = np.sign(actual)
    predicted_sign = np.sign(predicted)

    valid = predicted_sign != 0

    directional_accuracy = np.mean(
        actual_sign[valid]
        == predicted_sign[valid]
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy":
            directional_accuracy,
    }


# ============================================================
# BACKTEST METRICS
# ============================================================

def calculate_backtest_metrics(
    df: pd.DataFrame
) -> dict:

    strategy_returns = (
        df["strategy_return"]
        .to_numpy()
    )

    cumulative_return = (
        np.exp(
            strategy_returns.sum()
        ) - 1
    )

    observations = len(
        strategy_returns
    )

    years = (
        observations
        / 252
    )

    if years > 0:
        annualized_return = (
            np.exp(
                strategy_returns.sum()
                / years
            ) - 1
        )
    else:
        annualized_return = np.nan

    annualized_volatility = (
        strategy_returns.std(ddof=1)
        * np.sqrt(252)
    )

    if annualized_volatility == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (
            strategy_returns.mean()
            / strategy_returns.std(ddof=1)
            * np.sqrt(252)
        )

    equity_curve = np.exp(
        np.cumsum(strategy_returns)
    )

    running_max = np.maximum.accumulate(
        equity_curve
    )

    drawdowns = (
        equity_curve / running_max
        - 1
    )

    maximum_drawdown = drawdowns.min()

    win_rate = np.mean(
        strategy_returns > 0
    )

    if "position" in df.columns:
        positions = df["position"].to_numpy()

        trades = np.sum(
            positions[1:]
            != positions[:-1]
        )
    else:
        trades = np.nan

    return {
        "cumulative_return":
            cumulative_return,
        "annualized_return":
            annualized_return,
        "annualized_volatility":
            annualized_volatility,
        "sharpe_ratio":
            sharpe_ratio,
        "maximum_drawdown":
            maximum_drawdown,
        "win_rate":
            win_rate,
        "trades":
            trades,
        "observations":
            observations,
    }


# ============================================================
# BUILD FINAL TABLE
# ============================================================

def build_final_comparison():

    xgb_predictions = (
        load_walk_forward_predictions(
            "xgboost"
        )
    )

    lstm_predictions = (
        load_walk_forward_predictions(
            "lstm"
        )
    )

    xgb_backtest = load_backtest(
        "xgboost"
    )

    lstm_backtest = load_backtest(
        "lstm"
    )

    xgb_prediction_metrics = (
        calculate_prediction_metrics(
            xgb_predictions
        )
    )

    lstm_prediction_metrics = (
        calculate_prediction_metrics(
            lstm_predictions
        )
    )

    xgb_backtest_metrics = (
        calculate_backtest_metrics(
            xgb_backtest
        )
    )

    lstm_backtest_metrics = (
        calculate_backtest_metrics(
            lstm_backtest
        )
    )

    # Buy-and-hold
    buy_hold = xgb_backtest.copy()

    buy_hold["strategy_return"] = (
        buy_hold["actual_return"]
    )

    buy_hold["position"] = 1

    buy_hold_metrics = (
        calculate_backtest_metrics(
            buy_hold
        )
    )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    rows = []

    for model_name, prediction_metrics, backtest_metrics in [
        (
            "xgboost",
            xgb_prediction_metrics,
            xgb_backtest_metrics,
        ),
        (
            "lstm",
            lstm_prediction_metrics,
            lstm_backtest_metrics,
        ),
    ]:

        row = {
            "model": model_name,
            **prediction_metrics,
            **backtest_metrics,
        }

        rows.append(row)

    rows.append({
        "model": "buy_and_hold",
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "directional_accuracy": np.nan,
        **buy_hold_metrics,
    })

    results = pd.DataFrame(
        rows
    )

    return results


# ============================================================
# PLOT EQUITY CURVES
# ============================================================

def plot_equity_curves():

    xgb = load_backtest(
        "xgboost"
    )

    lstm = load_backtest(
        "lstm"
    )

    buy_hold = xgb.copy()

    xgb_equity = np.exp(
        xgb["strategy_return"].cumsum()
    )

    lstm_equity = np.exp(
        lstm["strategy_return"].cumsum()
    )

    buy_hold_equity = np.exp(
        buy_hold["actual_return"].cumsum()
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        xgb["date"],
        xgb_equity,
        label="XGBoost"
    )

    plt.plot(
        lstm["date"],
        lstm_equity,
        label="LSTM"
    )

    plt.plot(
        buy_hold["date"],
        buy_hold_equity,
        label="Buy & Hold"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.title(
        "Out-of-Sample Equity Curves"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"{TICKER}_equity_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Equity comparison saved to: "
        f"{output_path}"
    )


# ============================================================
# PLOT RISK / RETURN
# ============================================================

def plot_risk_return(
    results: pd.DataFrame
):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_data = results.dropna(
        subset=[
            "annualized_volatility",
            "annualized_return",
        ]
    )

    plt.figure(
        figsize=(8, 6)
    )

    for _, row in plot_data.iterrows():

        plt.scatter(
            row["annualized_volatility"],
            row["annualized_return"],
            s=100
        )

        plt.annotate(
            row["model"],
            (
                row["annualized_volatility"],
                row["annualized_return"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel(
        "Annualized volatility"
    )

    plt.ylabel(
        "Annualized return"
    )

    plt.title(
        "Risk–Return Comparison"
    )

    plt.grid(True)
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"{TICKER}_risk_return_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Risk-return comparison saved to: "
        f"{output_path}"
    )


# ============================================================
# PLOT PREDICTIONS
# ============================================================

def plot_prediction_comparison():

    xgb = load_walk_forward_predictions(
        "xgboost"
    )

    lstm = load_walk_forward_predictions(
        "lstm"
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        xgb["date"],
        xgb["actual_return"],
        label="Actual return",
        alpha=0.7
    )

    plt.plot(
        xgb["date"],
        xgb["predicted_return"],
        label="XGBoost prediction",
        alpha=0.7
    )

    plt.plot(
        lstm["date"],
        lstm["predicted_return"],
        label="LSTM prediction",
        alpha=0.7
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel("Date")
    plt.ylabel("Next-day log return")

    plt.title(
        "Walk-Forward Out-of-Sample Predictions"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"{TICKER}_prediction_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Prediction comparison saved to: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "PHASE 8 — FINAL MODEL COMPARISON"
    )
    print("=" * 60)

    results = build_final_comparison()

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        METRICS_DIR
        / f"{TICKER}_final_comparison.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        results.to_string(
            index=False
        )
    )

    print()

    print(
        f"Final comparison saved to: "
        f"{output_path}"
    )

    plot_equity_curves()

    plot_risk_return(
        results
    )

    plot_prediction_comparison()

    print()
    print("=" * 60)
    print("PHASE 8 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()