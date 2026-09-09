from pathlib import Path

import numpy as np
import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_clean_data(ticker: str) -> pd.DataFrame:
    """
    Load the cleaned market data.
    """
    file_path = PROCESSED_DIR / f"{ticker}_clean.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Clean data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Convert date back to datetime after reading CSV
    df["date"] = pd.to_datetime(df["date"])

    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial features and the next-day return target.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Log return
    # --------------------------------------------------

    df["log_return"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    # --------------------------------------------------
    # 2. Lagged returns
    # --------------------------------------------------

    df["return_lag_1"] = df["log_return"].shift(1)
    df["return_lag_2"] = df["log_return"].shift(2)
    df["return_lag_3"] = df["log_return"].shift(3)
    df["return_lag_5"] = df["log_return"].shift(5)
    df["return_lag_10"] = df["log_return"].shift(10)

    # --------------------------------------------------
    # 3. Moving averages
    # --------------------------------------------------

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    # Relative position of price to moving averages
    df["price_to_ma_5"] = df["close"] / df["ma_5"] - 1
    df["price_to_ma_20"] = df["close"] / df["ma_20"] - 1
    df["price_to_ma_50"] = df["close"] / df["ma_50"] - 1

    # --------------------------------------------------
    # 4. Rolling volatility
    # --------------------------------------------------

    df["volatility_5"] = df["log_return"].rolling(5).std()
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df["volatility_50"] = df["log_return"].rolling(50).std()

    # --------------------------------------------------
    # 5. Momentum
    # --------------------------------------------------

    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # --------------------------------------------------
    # 6. RSI
    # --------------------------------------------------

    df["rsi_14"] = calculate_rsi(df["close"], period=14)

    # --------------------------------------------------
    # 7. MACD
    # --------------------------------------------------

    ema_12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema_12 - ema_26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_histogram"] = (
        df["macd"] - df["macd_signal"]
    )

    # --------------------------------------------------
    # 8. Volume features
    # --------------------------------------------------

    df["volume_change"] = df["volume"].pct_change()

    df["volume_ma_20"] = df["volume"].rolling(20).mean()

    df["relative_volume"] = (
        df["volume"] / df["volume_ma_20"]
    )

    # --------------------------------------------------
    # 9. High-low price range
    # --------------------------------------------------

    df["daily_range"] = (
        df["high"] - df["low"]
    ) / df["close"]

    # --------------------------------------------------
    # 10. Open-close movement
    # --------------------------------------------------

    df["intraday_return"] = (
        df["close"] - df["open"]
    ) / df["open"]

    # --------------------------------------------------
    # 11. Prediction target
    # --------------------------------------------------

    df["target_return"] = np.log(
        df["close"].shift(-1) / df["close"]
    )

    # --------------------------------------------------
    # Remove rows created by rolling windows / lags
    # --------------------------------------------------

    df = df.dropna()

    # Ensure chronological ordering
    df = df.sort_values("date")

    # Reset index
    df = df.reset_index(drop=True)

    return df


def save_features(
    df: pd.DataFrame,
    ticker: str
) -> Path:
    """
    Save engineered features.
    """

    output_path = PROCESSED_DIR / f"{ticker}_features.csv"

    df.to_csv(
        output_path,
        index=False
    )

    return output_path


def process_features(ticker: str) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    """

    print(f"Loading cleaned {ticker} data...")

    df = load_clean_data(ticker)

    print(f"Clean observations: {len(df)}")

    df_features = create_features(df)

    print(
        f"Feature observations: {len(df_features)}"
    )

    output_path = save_features(
        df_features,
        ticker
    )

    print(
        f"Features saved to: {output_path}"
    )

    print(
        f"Number of columns: {len(df_features.columns)}"
    )

    return df_features


if __name__ == "__main__":
    process_features("AAPL")