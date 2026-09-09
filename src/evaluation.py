from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate the fraction of predictions
    with the correct return direction.
    """

    true_direction = np.sign(y_true)
    predicted_direction = np.sign(y_pred)

    return np.mean(
        true_direction == predicted_direction
    )


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Generic evaluation function.

    Parameters
    ----------
    y_true : actual target values
    y_pred : model predictions

    Returns
    -------
    Dictionary containing evaluation metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    directional_accuracy = (
        calculate_directional_accuracy(
            y_true,
            y_pred
        )
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy": directional_accuracy
    }


def save_metrics(
    metrics: dict,
    model_name: str,
    ticker: str
):
    """
    Save evaluation metrics to CSV.
    """

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.insert(
        0,
        "model",
        model_name
    )

    metrics_df.insert(
        0,
        "ticker",
        ticker
    )

    output_path = (
        METRICS_DIR
        / f"{ticker}_{model_name}_metrics.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False
    )

    return output_path


def save_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ticker: str
):
    """
    Save dates, actual returns and predictions.
    """

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not (
        len(dates)
        == len(y_true)
        == len(y_pred)
    ):
        raise ValueError(
            "dates, y_true and y_pred "
            "must have the same length."
        )

    predictions_df = pd.DataFrame({
        "date": dates.values,
        "actual_return": y_true,
        "predicted_return": y_pred
    })

    predictions_df["model"] = model_name
    predictions_df["ticker"] = ticker

    output_path = (
        PREDICTIONS_DIR
        / f"{ticker}_{model_name}_predictions.csv"
    )

    predictions_df.to_csv(
        output_path,
        index=False
    )

    return output_path


def evaluate_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ticker: str
):
    """
    Evaluate a model and save its metrics
    and predictions.
    """

    metrics = evaluate_model(
        y_true,
        y_pred
    )

    metrics_path = save_metrics(
        metrics,
        model_name,
        ticker
    )

    predictions_path = save_predictions(
        dates,
        y_true,
        y_pred,
        model_name,
        ticker
    )

    print(
        f"\n{'=' * 50}"
    )

    print(
        f"{model_name.upper()} EVALUATION"
    )

    print(
        f"{'=' * 50}"
    )

    print(
        f"Observations: "
        f"{len(y_true)}"
    )

    print(
        f"MAE: "
        f"{metrics['MAE']:.8f}"
    )

    print(
        f"RMSE: "
        f"{metrics['RMSE']:.8f}"
    )

    print(
        f"R²: "
        f"{metrics['R2']:.8f}"
    )

    print(
        f"Directional Accuracy: "
        f"{metrics['Directional Accuracy']:.4%}"
    )

    print(
        f"\nMetrics saved to:"
    )

    print(
        metrics_path
    )

    print(
        f"\nPredictions saved to:"
    )

    print(
        predictions_path
    )

    return metrics


def load_feature_data(
    ticker: str
) -> pd.DataFrame:
    """
    Load the engineered feature dataset.
    """

    file_path = (
        PROCESSED_DIR
        / f"{ticker}_features.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature data not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def evaluate_xgboost(
    df: pd.DataFrame,
    split_index: int,
    ticker: str
):
    """
    Train XGBoost and evaluate its test predictions.
    """

    from xgboost import XGBRegressor

    feature_columns = [
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

    X = df[feature_columns]
    y = df["target_return"]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    dates = df["date"].iloc[
        split_index:
    ].reset_index(drop=True)

    return evaluate_predictions(
        dates,
        y_test.values,
        predictions,
        "xgboost",
        ticker
    )


def evaluate_lstm(
    df: pd.DataFrame,
    split_index: int,
    ticker: str
):
    """
    Train LSTM and evaluate its test predictions.

    The LSTM uses 20-day sequences, so the first
    19 test observations cannot produce predictions
    using the current test-only sequence construction.
    """

    from lstm import (
        LSTMModel,
        create_sequences,
        train_lstm,
        predict_lstm,
        FEATURE_COLUMNS,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        DROPOUT,
        LEARNING_RATE,
        EPOCHS,
    )

    from sklearn.preprocessing import StandardScaler
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X = df[FEATURE_COLUMNS].values
    y = df["target_return"].values

    X_train_raw = X[:split_index]
    X_test_raw = X[split_index:]

    y_train_raw = y[:split_index]
    y_test_raw = y[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train_raw,
        SEQUENCE_LENGTH
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled,
        y_test_raw,
        SEQUENCE_LENGTH
    )

    X_train_tensor = torch.tensor(
        X_train_seq,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_seq,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_seq,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = LSTMModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    train_lstm(
        model,
        train_loader,
        EPOCHS,
        LEARNING_RATE
    )

    predictions = predict_lstm(
        model,
        X_test_tensor
    )

    # These dates correspond exactly to the
    # 524 LSTM predictions.
    dates = df["date"].iloc[
        split_index + SEQUENCE_LENGTH - 1:
    ].reset_index(drop=True)

    return evaluate_predictions(
        dates,
        y_test_seq,
        predictions,
        "lstm",
        ticker
    )


def main():
    ticker = "AAPL"

    print(
        f"Loading {ticker} feature data..."
    )

    df = load_feature_data(
        ticker
    )

    split_index = int(
        len(df) * 0.8
    )

    print(
        f"Total observations: {len(df)}"
    )

    print(
        f"Training observations: "
        f"{split_index}"
    )

    print(
        f"Testing observations: "
        f"{len(df) - split_index}"
    )

    print(
        "\nStarting XGBoost evaluation..."
    )

    xgboost_metrics = evaluate_xgboost(
        df,
        split_index,
        ticker
    )

    print(
        "\nStarting LSTM evaluation..."
    )

    lstm_metrics = evaluate_lstm(
        df,
        split_index,
        ticker
    )

    comparison = pd.DataFrame([
        {
            "model": "XGBoost",
            **xgboost_metrics
        },
        {
            "model": "LSTM",
            **lstm_metrics
        }
    ])

    print(
        f"\n{'=' * 70}"
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_path = (
        METRICS_DIR
        / f"{ticker}_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(
        f"\nComparison saved to:"
    )

    print(
        comparison_path
    )


if __name__ == "__main__":
    main()