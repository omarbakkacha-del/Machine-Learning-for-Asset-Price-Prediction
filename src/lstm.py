from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ============================================================
# Project directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 20
TRAIN_RATIO = 0.8

BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

LEARNING_RATE = 0.001
EPOCHS = 30

RANDOM_SEED = 42


# ============================================================
# Features
# ============================================================

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


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Load data
# ============================================================

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


# ============================================================
# Create sequences
# ============================================================

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int
):
    """
    Convert feature vectors into sequences.

    Each sample has the form:

        X[t-sequence_length+1 : t+1] -> y[t]

    Therefore the output has shape:

        (number_of_samples, sequence_length, number_of_features)
    """

    X_sequences = []
    y_sequences = []

    for i in range(sequence_length - 1, len(X)):

        X_sequences.append(
            X[i - sequence_length + 1:i + 1]
        )

        y_sequences.append(
            y[i]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# ============================================================
# LSTM model
# ============================================================

class LSTMModel(nn.Module):
    """
    LSTM neural network for next-day return prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.output_layer = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        # x shape:
        # (batch_size, sequence_length, input_size)

        lstm_output, _ = self.lstm(x)

        # Take the final time step
        last_output = lstm_output[:, -1, :]

        prediction = self.output_layer(
            last_output
        )

        return prediction.squeeze(-1)


# ============================================================
# Training
# ============================================================

def train_lstm(
    model,
    train_loader,
    epochs,
    learning_rate
):
    """
    Train the LSTM using mean squared error.
    """

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(X_batch)
            )

        average_loss = (
            total_loss
            / len(train_loader.dataset)
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"- Training loss: {average_loss:.8f}"
        )


# ============================================================
# Prediction
# ============================================================

def predict_lstm(
    model,
    X
):
    """
    Generate predictions from a trained LSTM.
    """

    model.eval()

    with torch.no_grad():

        predictions = model(X)

    return predictions.numpy()


# ============================================================
# Save model
# ============================================================

def save_lstm_model(
    model,
    ticker: str
):
    """
    Save trained LSTM parameters.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        MODELS_DIR
        / f"{ticker}_lstm.pt"
    )

    torch.save(
        model.state_dict(),
        output_path
    )

    return output_path


# ============================================================
# Complete LSTM pipeline
# ============================================================

def run_lstm(
    ticker: str = "AAPL"
):
    """
    Complete LSTM training pipeline.
    """

    print(
        f"Loading {ticker} feature data..."
    )

    df = load_feature_data(ticker)

    print(
        f"Observations: {len(df)}"
    )

    # --------------------------------------------------------
    # Construct X and y
    # --------------------------------------------------------

    X = df[FEATURE_COLUMNS].values
    y = df["target_return"].values

    print(
        f"Number of features: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Chronological train/test split
    # --------------------------------------------------------

    split_index = int(
        len(df) * TRAIN_RATIO
    )

    X_train_raw = X[:split_index]
    X_test_raw = X[split_index:]

    y_train_raw = y[:split_index]
    y_test_raw = y[split_index:]

    print(
        f"Training observations: "
        f"{len(X_train_raw)}"
    )

    print(
        f"Testing observations: "
        f"{len(X_test_raw)}"
    )

    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

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

    print(
        f"Training sequences: "
        f"{len(X_train_seq)}"
    )

    print(
        f"Testing sequences: "
        f"{len(X_test_seq)}"
    )

    print(
        f"Sequence shape: "
        f"{X_train_seq.shape[1:]}"
    )

    # --------------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------------

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

    y_test_tensor = torch.tensor(
        y_test_seq,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = LSTMModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    print(
        "\nLSTM architecture:"
    )

    print(model)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining LSTM..."
    )

    train_lstm(
        model,
        train_loader,
        EPOCHS,
        LEARNING_RATE
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = predict_lstm(
        model,
        X_test_tensor
    )

    print(
        f"Generated predictions: "
        f"{len(predictions)}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    output_path = save_lstm_model(
        model,
        ticker
    )

    print(
        f"Model saved to: {output_path}"
    )

    return (
        model,
        X_test_tensor,
        y_test_tensor,
        predictions
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_lstm("AAPL")