from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from XGBoost import train_xgboost, FEATURE_COLUMNS as XGB_FEATURE_COLUMNS
from lstm import (
    LSTMModel,
    create_sequences,
    predict_lstm,
    FEATURE_COLUMNS as LSTM_FEATURE_COLUMNS,
    BATCH_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS,
    SEQUENCE_LENGTH,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
WALK_FORWARD_DIR = RESULTS_DIR / "walk_forward"


TICKER = "AAPL"

# Initial amount of data used for the first training window.
INITIAL_TRAIN_RATIO = 0.60

# Number of observations predicted in each walk-forward fold.
TEST_WINDOW = 60

# Maximum number of folds.
MAX_FOLDS = None

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_feature_data(ticker: str) -> pd.DataFrame:
    """
    Load engineered feature data.
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

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    return df


def train_lstm_on_fold(
    X_train_raw,
    y_train_raw
):
    """
    Train an LSTM on one walk-forward
    training window.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train_raw,
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
        input_size=len(LSTM_FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

    return model, scaler


def predict_lstm_on_fold(
    model,
    scaler,
    X_train_raw,
    X_test_raw,
    y_test_raw
):
    """
    Generate predictions for the test window.

    The sequence for the first test observation
    uses the previous 19 observations from the
    training set.

    Therefore every test observation receives
    exactly one prediction.
    """

    X_combined_raw = np.concatenate(
        [
            X_train_raw,
            X_test_raw
        ],
        axis=0
    )

    X_combined_scaled = scaler.transform(
        X_combined_raw
    )

    X_test_sequences = []

    for i in range(
        len(X_train_raw),
        len(X_combined_scaled)
    ):

        sequence_start = (
            i - SEQUENCE_LENGTH + 1
        )

        sequence = X_combined_scaled[
            sequence_start:i + 1
        ]

        X_test_sequences.append(
            sequence
        )

    X_test_sequences = np.array(
        X_test_sequences
    )

    X_test_tensor = torch.tensor(
        X_test_sequences,
        dtype=torch.float32
    )

    predictions = predict_lstm(
        model,
        X_test_tensor
    )

    return predictions


def run_xgboost_fold(
    X_train,
    y_train,
    X_test
):
    """
    Train XGBoost and generate predictions
    for one test window.
    """

    model = train_xgboost(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    return predictions


def run_walk_forward(
    df: pd.DataFrame,
    model_name: str
):
    """
    Perform expanding-window walk-forward validation.
    """

    if model_name == "xgboost":
        feature_columns = XGB_FEATURE_COLUMNS

    elif model_name == "lstm":
        feature_columns = LSTM_FEATURE_COLUMNS

    else:
        raise ValueError(
            "model_name must be 'xgboost' or 'lstm'."
        )

    X = df[
        feature_columns
    ].values

    y = df[
        "target_return"
    ].values

    dates = df[
        "date"
    ]

    initial_train_size = int(
        len(df)
        * INITIAL_TRAIN_RATIO
    )

    results = []

    train_end = initial_train_size

    fold = 1

    while (
    train_end < len(df)
    and (
        MAX_FOLDS is None
        or fold <= MAX_FOLDS)
    ):

        test_start = train_end

        test_end = min(
            test_start + TEST_WINDOW,
            len(df)
        )

        X_train = X[
            :train_end
        ]

        y_train = y[
            :train_end
        ]

        X_test = X[
            test_start:test_end
        ]

        y_test = y[
            test_start:test_end
        ]

        test_dates = dates.iloc[
            test_start:test_end
        ].reset_index(drop=True)

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"{model_name.upper()} "
            f"FOLD {fold}"
        )

        print(
            "=" * 70
        )

        print(
            f"Training period: "
            f"{dates.iloc[0].date()} "
            f"-> "
            f"{dates.iloc[train_end - 1].date()}"
        )

        print(
            f"Testing period: "
            f"{dates.iloc[test_start].date()} "
            f"-> "
            f"{dates.iloc[test_end - 1].date()}"
        )

        print(
            f"Training observations: "
            f"{len(X_train)}"
        )

        print(
            f"Testing observations: "
            f"{len(X_test)}"
        )

        if model_name == "xgboost":

            predictions = run_xgboost_fold(
                X_train,
                y_train,
                X_test
            )

        else:

            model, scaler = train_lstm_on_fold(
                X_train,
                y_train
            )

            predictions = predict_lstm_on_fold(
                model,
                scaler,
                X_train,
                X_test,
                y_test
            )

        print(
            f"Generated predictions: "
            f"{len(predictions)}"
        )

        fold_df = pd.DataFrame({
            "date": test_dates,
            "actual_return": y_test,
            "predicted_return": predictions,
            "model": model_name,
            "fold": fold
        })

        results.append(
            fold_df
        )

        train_end = test_end

        fold += 1

    if not results:
        raise RuntimeError(
            "No walk-forward folds were created."
        )

    return pd.concat(
        results,
        ignore_index=True
    )


def calculate_metrics(
    predictions_df: pd.DataFrame
):
    """
    Calculate aggregate walk-forward metrics.
    """

    y_true = predictions_df[
        "actual_return"
    ].values

    y_pred = predictions_df[
        "predicted_return"
    ].values

    mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    directional_accuracy = np.mean(
        np.sign(y_true)
        == np.sign(y_pred)
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional Accuracy":
            directional_accuracy
    }


def save_results(
    predictions_df: pd.DataFrame,
    metrics: dict,
    model_name: str
):
    """
    Save walk-forward predictions and metrics.
    """

    WALK_FORWARD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_path = (
        WALK_FORWARD_DIR
        / f"{TICKER}_{model_name}_predictions.csv"
    )

    metrics_path = (
        WALK_FORWARD_DIR
        / f"{TICKER}_{model_name}_metrics.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    metrics_df = pd.DataFrame([
        {
            "model": model_name,
            **metrics
        }
    ])

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    return (
        predictions_path,
        metrics_path
    )


def main():

    print(
        f"Loading {TICKER} feature data..."
    )

    df = load_feature_data(
        TICKER
    )

    print(
        f"Total observations: "
        f"{len(df)}"
    )

    print(
        f"Initial training size: "
        f"{int(len(df) * INITIAL_TRAIN_RATIO)}"
    )

    print(
        f"Test window: "
        f"{TEST_WINDOW}"
    )

    print(
        f"Maximum folds: "
        f"{MAX_FOLDS}"
    )

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    print(
        "\n\n"
        "Starting XGBoost walk-forward validation..."
    )

    xgb_predictions = run_walk_forward(
        df,
        "xgboost"
    )

    xgb_metrics = calculate_metrics(
        xgb_predictions
    )

    xgb_paths = save_results(
        xgb_predictions,
        xgb_metrics,
        "xgboost"
    )

    print(
        "\nXGBoost walk-forward results:"
    )

    for metric, value in xgb_metrics.items():

        if metric == "Directional Accuracy":

            print(
                f"{metric}: "
                f"{value:.4%}"
            )

        else:

            print(
                f"{metric}: "
                f"{value:.8f}"
            )

    # --------------------------------------------------
    # LSTM
    # --------------------------------------------------

    print(
        "\n\n"
        "Starting LSTM walk-forward validation..."
    )

    lstm_predictions = run_walk_forward(
        df,
        "lstm"
    )

    lstm_metrics = calculate_metrics(
        lstm_predictions
    )

    lstm_paths = save_results(
        lstm_predictions,
        lstm_metrics,
        "lstm"
    )

    print(
        "\nLSTM walk-forward results:"
    )

    for metric, value in lstm_metrics.items():

        if metric == "Directional Accuracy":

            print(
                f"{metric}: "
                f"{value:.4%}"
            )

        else:

            print(
                f"{metric}: "
                f"{value:.8f}"
            )

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    comparison = pd.DataFrame([
        {
            "model": "XGBoost",
            **xgb_metrics
        },
        {
            "model": "LSTM",
            **lstm_metrics
        }
    ])

    comparison_path = (
        WALK_FORWARD_DIR
        / f"{TICKER}_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "WALK-FORWARD MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        WALK_FORWARD_DIR
    )


if __name__ == "__main__":
    main()