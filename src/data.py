from pathlib import Path
import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(ticker: str) -> pd.DataFrame:
    """
    Load raw market data for a given ticker.
    """
    file_path = RAW_DIR / f"{ticker}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    df = pd.read_csv(file_path)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean historical market data downloaded from yfinance.
    """

    # Handle the CSV format produced by yfinance when downloading
    # a single ticker.
    if "Price" in df.columns:

        # The first row contains "Ticker"
        # The second row contains "Date"
        # Remove these two metadata rows.
        if len(df) >= 2 and str(df.iloc[0]["Price"]).lower() == "ticker":
            df = df.iloc[2:].copy()

        # Rename the Price column to Date
        df = df.rename(columns={"Price": "Date"})

    # Standardize column names
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    # Convert Date to datetime
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Convert market columns to numeric
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove rows with invalid dates
    df = df.dropna(subset=["date"])

    # Remove rows with missing prices
    price_columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    existing_price_columns = [
        column for column in price_columns
        if column in df.columns
    ]

    df = df.dropna(subset=existing_price_columns)

    # Remove impossible/non-positive prices
    for column in existing_price_columns:
        df = df[df[column] > 0]

    # Sort chronologically
    df = df.sort_values("date")

    # Remove duplicate dates
    df = df.drop_duplicates(subset=["date"])

    # Reset index
    df = df.reset_index(drop=True)

    return df


def save_processed_data(df: pd.DataFrame, ticker: str) -> Path:
    """
    Save cleaned data to the processed directory.
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / f"{ticker}_clean.csv"

    df.to_csv(output_path, index=False)

    return output_path


def process_ticker(ticker: str) -> pd.DataFrame:
    """
    Complete pipeline:
    load raw data → clean → save processed data.
    """

    print(f"Loading raw {ticker} data...")

    df = load_raw_data(ticker)

    print(f"Raw observations: {len(df)}")

    df_clean = clean_data(df)

    print(f"Clean observations: {len(df_clean)}")

    output_path = save_processed_data(df_clean, ticker)

    print(f"Processed data saved to: {output_path}")

    return df_clean


if __name__ == "__main__":
    process_ticker("AAPL")