from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WALK_FORWARD_DIR = PROJECT_ROOT / "results" / "walk_forward"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TICKER = "AAPL"

# Transaction cost per position change.
# 0.001 = 0.10% per trade.
TRANSACTION_COST = 0.001

TRADING_DAYS_PER_YEAR = 252


def load_predictions(ticker: str, model_name: str) -> pd.DataFrame:
    """
    Load out-of-sample predictions generated during
    walk-forward validation.
    """

    file_path = (
        WALK_FORWARD_DIR
        / f"{ticker}_{model_name}_predictions.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return df.sort_values("date").reset_index(drop=True)


def create_strategy_returns(
    predictions: pd.DataFrame,
    transaction_cost: float
) -> pd.DataFrame:
    """
    Convert model predictions into trading positions
    and calculate daily strategy returns.
    """

    df = predictions.copy()

    # Long if predicted return > 0.
    # Short if predicted return < 0.
    df["position"] = np.sign(df["predicted_return"])

    # If prediction is exactly zero, remain out of the market.
    df.loc[
        df["predicted_return"] == 0,
        "position"
    ] = 0

    # Position change determines turnover.
    df["position_change"] = (
        df["position"]
        .diff()
        .abs()
        .fillna(df["position"].abs())
    )

    # Transaction cost.
    df["transaction_cost"] = (
        df["position_change"] * transaction_cost
    )

    # Gross strategy return.
    df["gross_strategy_return"] = (
        df["position"] * df["actual_return"]
    )

    # Net strategy return.
    df["strategy_return"] = (
        df["gross_strategy_return"]
        - df["transaction_cost"]
    )

    # Buy-and-hold return.
    df["buy_hold_return"] = df["actual_return"]

    # Cumulative wealth.
    df["strategy_wealth"] = (
        np.exp(df["strategy_return"].cumsum())
    )

    df["buy_hold_wealth"] = (
        np.exp(df["buy_hold_return"].cumsum())
    )

    return df


def calculate_metrics(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR
) -> dict:

    returns = returns.dropna()

    if len(returns) == 0:
        raise ValueError("No returns available.")

    cumulative_return = (
        np.exp(returns.sum()) - 1
    )

    annualized_return = (
        np.exp(
            returns.mean() * trading_days
        ) - 1
    )

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(trading_days)
    )

    if annualized_volatility > 0:
        sharpe_ratio = (
            returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(trading_days)
        )
    else:
        sharpe_ratio = np.nan

    cumulative_wealth = np.exp(
        returns.cumsum()
    )

    running_max = cumulative_wealth.cummax()

    drawdown = (
        cumulative_wealth / running_max
        - 1
    )

    maximum_drawdown = drawdown.min()

    positive_returns = (
        returns > 0
    ).sum()

    win_rate = (
        positive_returns / len(returns)
    )

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
        "win_rate": win_rate,
        "observations": len(returns),
    }


def count_trades(df: pd.DataFrame) -> int:
    """
    Count the number of position changes.
    """

    return int(
        (df["position_change"] > 0).sum()
    )


def backtest_model(
    ticker: str,
    model_name: str
):

    print(
        f"\nBacktesting {model_name.upper()}..."
    )

    predictions = load_predictions(
        ticker,
        model_name
    )

    print(
        f"Out-of-sample observations: "
        f"{len(predictions)}"
    )

    df = create_strategy_returns(
        predictions,
        TRANSACTION_COST
    )

    strategy_metrics = calculate_metrics(
        df["strategy_return"]
    )

    buy_hold_metrics = calculate_metrics(
        df["buy_hold_return"]
    )

    strategy_metrics["trades"] = count_trades(df)

    buy_hold_metrics["trades"] = 0

    strategy_metrics["model"] = model_name
    buy_hold_metrics["model"] = "buy_and_hold"

    return (
        df,
        strategy_metrics,
        buy_hold_metrics
    )


def save_results(
    df: pd.DataFrame,
    model_name: str
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_DIR
        / "predictions"
        / f"{TICKER}_{model_name}_backtest.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Backtest data saved to: {output_path}"
    )


def plot_equity_curves(
    xgb_df: pd.DataFrame,
    lstm_df: pd.DataFrame
):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        xgb_df["date"],
        xgb_df["strategy_wealth"],
        label="XGBoost"
    )

    plt.plot(
        lstm_df["date"],
        lstm_df["strategy_wealth"],
        label="LSTM"
    )

    plt.plot(
        xgb_df["date"],
        xgb_df["buy_hold_wealth"],
        label="Buy and Hold"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title(
        f"{TICKER} Out-of-Sample Backtest"
    )

    plt.legend()
    plt.grid(True)

    output_path = (
        FIGURES_DIR
        / f"{TICKER}_backtest_equity.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Equity curve saved to: {output_path}"
    )


def main():

    print("=" * 60)
    print("PHASE 7 — BACKTESTING")
    print("=" * 60)

    xgb_df, xgb_metrics, buy_hold_xgb = (
        backtest_model(
            TICKER,
            "xgboost"
        )
    )

    lstm_df, lstm_metrics, buy_hold_lstm = (
        backtest_model(
            TICKER,
            "lstm"
        )
    )

    # Verify that both models use exactly
    # the same out-of-sample dates.
    if not xgb_df["date"].equals(
        lstm_df["date"]
    ):
        raise ValueError(
            "XGBoost and LSTM dates do not match."
        )

    save_results(
        xgb_df,
        "xgboost"
    )

    save_results(
        lstm_df,
        "lstm"
    )

    plot_equity_curves(
        xgb_df,
        lstm_df
    )

    comparison = pd.DataFrame([
        xgb_metrics,
        lstm_metrics,
        buy_hold_xgb
    ])

    metrics_path = (
        RESULTS_DIR
        / "metrics"
        / f"{TICKER}_backtest_metrics.csv"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nMetrics saved to: {metrics_path}"
    )

    print("\nBACKTEST RESULTS")
    print("=" * 60)

    print(
        comparison[
            [
                "model",
                "cumulative_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
                "win_rate",
                "trades",
                "observations",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()