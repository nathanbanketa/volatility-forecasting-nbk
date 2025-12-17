import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import build_spy_pipeline
from src.models.garch import garch_model
from src.models.lstm import LSTMVolatility


def backtest_models(
    csv_path: str,
    model_path: str,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20,
    device: str = None
):
    """
    Backtest GARCH and LSTM models on test data.
    
    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq = build_spy_pipeline(
        csv_path=str(csv_path),
        split_date=split_date,
        vol_window=vol_window,
        seq_window=seq_window,
    )

    X_test_seq = np.array(X_test_seq)
    y_test_seq = np.array(y_test_seq).reshape(-1)

    returns_full = pd.concat([train_df["ret"], test_df["ret"]])
    res, garch_vol_full = garch_model(returns_full, p=1, q=1)
    garch_vol_full.index = returns_full.index

    garch_vol_test = garch_vol_full.loc[test_df.index]
    realized_vol_test = test_df["realized_vol"]

    X_train_seq = np.array(X_train_seq)
    feature_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
    feature_std = X_train_seq.std(axis=(0, 1), keepdims=True) + 1e-8
    X_test_seq_norm = (X_test_seq - feature_mean) / feature_std

    X_test_t = torch.tensor(X_test_seq_norm, dtype=torch.float32).to(device)

    input_size = X_test_seq.shape[-1]
    model = LSTMVolatility(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    state_path = Path(model_path)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    with torch.no_grad():
        lstm_pred = model(X_test_t).cpu().numpy()

    num_test_rows = len(test_df)
    num_seqs = X_test_seq.shape[0]
    offset = num_test_rows - num_seqs

    seq_index = test_df.index[offset:]

    realized_seq = realized_vol_test.iloc[offset:].values
    garch_seq = garch_vol_test.iloc[offset:].values
    lstm_seq = lstm_pred

    returns_seq = test_df["ret"].iloc[offset:].values

    print("Lengths:",
          "seq_index", len(seq_index),
          "realized_seq", realized_seq.shape[0],
          "garch_seq", garch_seq.shape[0],
          "lstm_seq", lstm_seq.shape[0],
          "returns_seq", returns_seq.shape[0])

    df_compare = pd.DataFrame(
        {
            "realized_vol": realized_seq,
            "garch_vol": garch_seq,
            "lstm_vol": lstm_seq,
            "ret": returns_seq,
        },
        index=seq_index,
    )

    garch_rmse = np.sqrt(mean_squared_error(df_compare["realized_vol"], df_compare["garch_vol"]))
    lstm_rmse = np.sqrt(mean_squared_error(df_compare["realized_vol"], df_compare["lstm_vol"]))

    print("=== OoS RMSE on aligned test sequences ===")
    print("GARCH RMSE:", garch_rmse)
    print("LSTM v2 RMSE:", lstm_rmse)

    alpha = 0.99
    z_99 = 2.33

    garch_var_99 = -z_99 * df_compare["garch_vol"].values
    lstm_var_99 = -z_99 * df_compare["lstm_vol"].values

    returns = df_compare["ret"].values

    garch_breaches_99 = np.mean(returns < garch_var_99)
    lstm_breaches_99 = np.mean(returns < lstm_var_99)

    print("\n=== 99% VaR breach rates (expected ~1%) ===")
    print("GARCH 99% VaR breach rate:", garch_breaches_99)
    print("LSTM v2 99% VaR breach rate:", lstm_breaches_99)

    realized = df_compare["realized_vol"].values
    high_cutoff = np.percentile(realized, 90)
    mask_high = realized > high_cutoff

    garch_rmse_high = np.sqrt(mean_squared_error(realized[mask_high], garch_seq[mask_high]))
    lstm_rmse_high = np.sqrt(mean_squared_error(realized[mask_high], lstm_seq[mask_high]))

    print("\n=== High-vol regime (top 10% realized vol) RMSE ===")
    print("GARCH RMSE (high vol):", garch_rmse_high)
    print("LSTM v2 RMSE (high vol):", lstm_rmse_high)
    
    return {
        "df_compare": df_compare,
        "garch_rmse": garch_rmse,
        "lstm_rmse": lstm_rmse,
        "garch_breaches_99": garch_breaches_99,
        "lstm_breaches_99": lstm_breaches_99,
        "garch_rmse_high": garch_rmse_high,
        "lstm_rmse_high": lstm_rmse_high,
    }

