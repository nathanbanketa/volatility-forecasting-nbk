import pandas as pd
import numpy as np
from typing import Tuple


def load_spy_data(csv_path: str) -> pd.DataFrame:
    """
    Load SPY data from a CSV file and return a DataFrame
    indexed by Date, sorted in ascending order.

    Expected columns:
      - Either a price column like 'Adj Close' or 'Close'
      - Or a returns column that includes 'ret' in the name
    """
    df = pd.read_csv(csv_path)

    if 'Date' not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a 'ret' column with daily log returns.

    If a 'ret' column already exists, it is preserved.
    Otherwise, the function looks for a price column
    such as 'Adj Close' or 'Close' and computes:
        ret_t = log(P_t / P_{t-1})
    """
    if 'ret' in df.columns:
        # Already has returns
        return df

    # Try to find a price column
    price_col = None
    for col in df.columns:
        if col.lower() in ['adj close', 'adj_close', 'close']:
            price_col = col
            break

    if price_col is None:
        raise ValueError("No 'ret' column found and no price column like 'Adj Close' or 'Close' present.")

    df['ret'] = np.log(df[price_col] / df[price_col].shift(1))
    df = df.dropna(subset=['ret'])

    return df


def add_realized_vol(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Add a realized volatility column 'realized_vol' using a rolling
    standard deviation of returns over a given window length (in days).
    """
    if 'ret' not in df.columns:
        raise ValueError("DataFrame must have a 'ret' column before computing realized volatility.")

    df['realized_vol'] = df['ret'].rolling(window).std()
    df = df.dropna(subset=['realized_vol'])
    return df


def train_test_split(
    df: pd.DataFrame,
    split_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train and test sets based on a split date.

    All rows with Date < split_date go to the training set.
    All rows with Date >= split_date go to the test set.
    """
    train_df = df.loc[df.index < split_date].copy()
    test_df = df.loc[df.index >= split_date].copy()

    return train_df, test_df


def make_windowed_dataset(
    returns: np.ndarray,
    vols: np.ndarray,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create windowed datasets for sequence models.

    Inputs:
      returns: 1D numpy array of daily returns (length N)
      vols:    1D numpy array of realized volatility (length N)
      window_size: number of past days to use as input

    Output:
      X: shape (num_samples, window_size, 1)
      y: shape (num_samples,)
    """
    X_list = []
    y_list = []

    for i in range(window_size, len(returns)):
        window = returns[i - window_size:i]
        X_list.append(window.reshape(-1, 1))  # (window_size, 1)
        y_list.append(vols[i])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def build_spy_pipeline(
    csv_path: str,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20
):
    """
    High-level helper that:
      1. Loads SPY data from CSV
      2. Adds returns
      3. Adds realized volatility
      4. Splits into train/test
      5. Builds windowed datasets for future DL models

    Returns:
      train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq
    """
    df = load_spy_data(csv_path)
    df = add_returns(df)
    df = add_realized_vol(df, window=vol_window)

    train_df, test_df = train_test_split(df, split_date)

    train_returns = train_df['ret'].values
    train_vols = train_df['realized_vol'].values
    test_returns = test_df['ret'].values
    test_vols = test_df['realized_vol'].values

    X_train_seq, y_train_seq = make_windowed_dataset(train_returns, train_vols, seq_window)
    X_test_seq, y_test_seq = make_windowed_dataset(test_returns, test_vols, seq_window)

    return train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq
