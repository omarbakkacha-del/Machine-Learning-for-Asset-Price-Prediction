from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"


# Features used by the model
FEATURE_COLUMNS = [
    "log_return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_50",
    "price_to_ma_5",
    "price_to_ma_20",
    "price_to_ma_50",
    "volatility_5",
    "volatility_20",
    "volatility_50",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volume_change",
    "volume_ma_20",
    "relative_volume",
    "daily_range",
    "intraday_return",
]


def load_feature_data(ticker: str) -> pd.DataFrame:
    """
    Load engineered feature data.
    """

    file_path = PROCESSED_DIR / f"{ticker}_features.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature data not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return df


def prepare_xy(df: pd.DataFrame):
    """
    Construct the feature matrix X and target vector y.

    X_t = information available at time t
    y_t = next-day log return
    """

    X = df[FEATURE_COLUMNS]
    y = df["target_return"]

    return X, y


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series
):
    """
    Train an XGBoost regression model.
    """

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, ticker: str):
    """
    Save trained XGBoost model.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = MODELS_DIR / f"{ticker}_xgboost.json"

    model.save_model(output_path)

    return output_path


def run_xgboost(ticker: str = "AAPL"):
    """
    Complete XGBoost training pipeline.
    """

    print(f"Loading {ticker} feature data...")

    df = load_feature_data(ticker)

    print(f"Observations: {len(df)}")

    X, y = prepare_xy(df)

    print(f"Number of features: {X.shape[1]}")
    print(f"Number of observations: {X.shape[0]}")

    # Chronological train/test split.
    # We deliberately do NOT shuffle financial time-series data.
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    # Train model
    print("Training XGBoost model...")

    model = train_xgboost(
        X_train,
        y_train
    )

    print("Training complete.")

    # Generate predictions
    predictions = model.predict(X_test)

    print(f"Generated predictions: {len(predictions)}")

    # Save model
    output_path = save_model(
        model,
        ticker
    )

    print(f"Model saved to: {output_path}")

    return model, X_test, y_test, predictions


if __name__ == "__main__":
    run_xgboost("AAPL")