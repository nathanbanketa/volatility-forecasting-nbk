import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, List


def _load_returns_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"Returns": "ret"})
    return df


def _fetch_vix_series(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    vix = yf.download("^VIX", start=start_date, end=end_date + pd.Timedelta(days=1))
    vix_series = vix["Close"].astype(float)
    vix_series.index = pd.to_datetime(vix_series.index)
    return vix_series


def build_spy_pipeline(
    csv_path: str,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20,
    use_vix: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    csv_path = str(csv_path)
    df = _load_returns_csv(csv_path)
    df["realized_vol"] = df["ret"].rolling(vol_window).std()
    df["abs_ret"] = df["ret"].abs()
    df["ret_sq"] = df["ret"] ** 2
    df["realized_vol_lag1"] = df["realized_vol"].shift(1)

    if use_vix:
        start_date = df.index.min()
        end_date = df.index.max()
        vix_series = _fetch_vix_series(start_date, end_date)

        vix_aligned = vix_series.reindex(df.index)
        df["vix"] = vix_aligned.ffill()

    df = df.dropna()

    feature_cols: List[str] = ["ret", "abs_ret", "ret_sq", "realized_vol_lag1"]
    if use_vix:
        feature_cols.append("vix")

    target_col = "realized_vol"

    split_ts = pd.to_datetime(split_date)
    train_df = df[df.index < split_ts].copy()
    test_df = df[df.index >= split_ts].copy()

    if len(train_df) <= seq_window or len(test_df) <= seq_window:
        raise ValueError("Not enough data in train or test after split for the given seq_window.")

    def make_sequences(section: pd.DataFrame):
        X_list = []
        y_list = []
        data = section[feature_cols]
        target = section[target_col]
        for i in range(seq_window, len(section)):
            window = data.iloc[i - seq_window:i].values
            y_val = target.iloc[i]
            X_list.append(window)
            y_list.append(y_val)
        return np.array(X_list), np.array(y_list)

    X_train_seq, y_train_seq = make_sequences(train_df)
    X_test_seq, y_test_seq = make_sequences(test_df)

    return train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq


def download_spy_returns():
    """Download SPY returns data and save to CSV."""
    ticker = "SPY"
    start_date = "2010-01-01"
    from datetime import datetime
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")

    spy = yf.Ticker(ticker)
    hist = spy.history(start=start_date, end=end_date)

    returns = hist['Close'].pct_change()

    df = pd.DataFrame({
        'Date': returns.index,
        'Returns': returns.values
    })

    df = df.dropna()

    df['Date'] = pd.to_datetime(df['Date']).dt.date

    df = df.reset_index(drop=True)

    output_file = "SPY_returns.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Data saved to {output_file}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df
