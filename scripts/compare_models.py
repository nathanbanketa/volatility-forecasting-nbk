"""
Compare GARCH and LSTM models with different evaluation strategies.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import build_spy_pipeline
from src.models.garch import garch_model
from src.models.lstm import LSTMVolatility


def compare_models_2020_cutoff(
    csv_path: str = None,
    model_path: str = None,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20,
    device: str = None,
    show_plots: bool = True,
):
    """
    Compare models using a 2020 cutoff date for train/test split.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    if csv_path is None:
        csv_path = project_root / "data" / "SPY_returns.csv"
    else:
        csv_path = Path(csv_path)
    
    if model_path is None:
        model_path = project_root / "models" / "lstm_v2_best.pt"
    else:
        model_path = Path(model_path)
    
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
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

    input_size = X_test_seq.shape[-1]
    model = LSTMVolatility(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
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

    print(len(seq_index), realized_seq.shape[0], garch_seq.shape[0], lstm_seq.shape[0])

    df_compare = pd.DataFrame(
        {
            "realized_vol": realized_seq,
            "garch_vol": garch_seq,
            "lstm_vol": lstm_seq,
        },
        index=seq_index,
    )

    garch_rmse = np.sqrt(mean_squared_error(df_compare["realized_vol"], df_compare["garch_vol"]))
    lstm_rmse = np.sqrt(mean_squared_error(df_compare["realized_vol"], df_compare["lstm_vol"]))

    print("Aligned GARCH Test RMSE:", garch_rmse)
    print("Aligned LSTM v2 Test RMSE:", lstm_rmse)

    if show_plots:
        plt.figure(figsize=(12, 5))
        plt.plot(df_compare.index, df_compare["realized_vol"], label="Realized Vol", alpha=0.7)
        plt.plot(df_compare.index, df_compare["garch_vol"], label="GARCH Vol", alpha=0.7)
        plt.plot(df_compare.index, df_compare["lstm_vol"], label="LSTM v2 Vol", alpha=0.7)
        plt.title("Volatility Forecasts vs Realized Volatility (Test Set)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.scatter(df_compare["realized_vol"], df_compare["garch_vol"],
                    alpha=0.4, label="GARCH", color="blue")
        plt.scatter(df_compare["realized_vol"], df_compare["lstm_vol"],
                    alpha=0.4, label="LSTM v2", color="orange")
        plt.plot([df_compare["realized_vol"].min(), df_compare["realized_vol"].max()],
                 [df_compare["realized_vol"].min(), df_compare["realized_vol"].max()],
                 'k--', label="Perfect Fit")
        plt.title("Predicted vs Actual (Scatter Plot)")
        plt.xlabel("Realized Vol")
        plt.ylabel("Predicted Vol")
        plt.legend()
        plt.tight_layout()
        plt.show()

        garch_errors = df_compare["realized_vol"] - df_compare["garch_vol"]
        lstm_errors = df_compare["realized_vol"] - df_compare["lstm_vol"]

        plt.figure(figsize=(12,5))
        plt.hist(garch_errors, bins=40, alpha=0.5, label="GARCH Errors")
        plt.hist(lstm_errors, bins=40, alpha=0.5, label="LSTM v2 Errors")
        plt.title("Residual Distribution Comparison")
        plt.xlabel("Error (Realized - Predicted)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return df_compare, garch_rmse, lstm_rmse


def compare_models_random_oos(
    csv_path: str = None,
    model_path: str = None,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20,
    n_sample: int = 100,
    device: str = None,
    show_plots: bool = True,
):
    """
    Compare models using a random out-of-sample split.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    if csv_path is None:
        csv_path = project_root / "data" / "SPY_returns.csv"
    else:
        csv_path = Path(csv_path)
    
    if model_path is None:
        model_path = project_root / "models" / "lstm_v2_best.pt"
    else:
        model_path = Path(model_path)
    
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
    y_test_t = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

    input_size = X_test_seq.shape[-1]
    model = LSTMVolatility(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
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

    df_compare = pd.DataFrame(
        {
            "realized_vol": realized_seq,
            "garch_vol": garch_seq,
            "lstm_vol": lstm_seq,
        },
        index=seq_index,
    )

    rng = np.random.default_rng(42)
    if n_sample > len(df_compare):
        n_sample = len(df_compare)

    sample_indices = rng.choice(len(df_compare), size=n_sample, replace=False)
    df_sample = df_compare.iloc[sample_indices].sort_index()

    garch_rmse_sample = np.sqrt(
        mean_squared_error(df_sample["realized_vol"], df_sample["garch_vol"])
    )
    lstm_rmse_sample = np.sqrt(
        mean_squared_error(df_sample["realized_vol"], df_sample["lstm_vol"])
    )

    print(f"Random OoS subset size: {len(df_sample)}")
    print("Random-OoS GARCH RMSE:", garch_rmse_sample)
    print("Random-OoS LSTM v2 RMSE:", lstm_rmse_sample)

    if show_plots:
        plt.figure(figsize=(12, 5))
        plt.plot(df_sample.index, df_sample["realized_vol"], label="Realized Vol", alpha=0.7)
        plt.plot(df_sample.index, df_sample["garch_vol"], label="GARCH Vol", alpha=0.7)
        plt.plot(df_sample.index, df_sample["lstm_vol"], label="LSTM v2 Vol", alpha=0.7)
        plt.title("Volatility Forecasts vs Realized Volatility (Random OoS Subset)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.scatter(df_sample["realized_vol"], df_sample["garch_vol"],
                    alpha=0.4, label="GARCH", color="blue")
        plt.scatter(df_sample["realized_vol"], df_sample["lstm_vol"],
                    alpha=0.4, label="LSTM v2", color="orange")
        plt.plot(
            [df_sample["realized_vol"].min(), df_sample["realized_vol"].max()],
            [df_sample["realized_vol"].min(), df_sample["realized_vol"].max()],
            'k--', label="Perfect Fit"
        )
        plt.title("Predicted vs Actual (Random OoS Subset)")
        plt.xlabel("Realized Vol")
        plt.ylabel("Predicted Vol")
        plt.legend()
        plt.tight_layout()
        plt.show()

        garch_errors = df_sample["realized_vol"] - df_sample["garch_vol"]
        lstm_errors = df_sample["realized_vol"] - df_sample["lstm_vol"]

        plt.figure(figsize=(12, 5))
        plt.hist(garch_errors, bins=30, alpha=0.5, label="GARCH Errors")
        plt.hist(lstm_errors, bins=30, alpha=0.5, label="LSTM v2 Errors")
        plt.title("Residual Distribution Comparison (Random OoS Subset)")
        plt.xlabel("Error (Realized - Predicted)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return df_sample, garch_rmse_sample, lstm_rmse_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GARCH and LSTM models")
    parser.add_argument("--method", type=str, default="2020_cutoff", 
                        choices=["2020_cutoff", "random_oos"],
                        help="Comparison method: 2020_cutoff or random_oos")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV file with returns data")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained LSTM model")
    parser.add_argument("--split-date", type=str, default="2020-01-01",
                        help="Date to split train/test")
    parser.add_argument("--n-sample", type=int, default=100,
                        help="Number of samples for random OoS (only for random_oos method)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Don't show plots")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    if args.method == "2020_cutoff":
        compare_models_2020_cutoff(
            csv_path=args.csv_path,
            model_path=args.model_path,
            split_date=args.split_date,
            device=args.device,
            show_plots=not args.no_plots,
        )
    else:
        compare_models_random_oos(
            csv_path=args.csv_path,
            model_path=args.model_path,
            split_date=args.split_date,
            n_sample=args.n_sample,
            device=args.device,
            show_plots=not args.no_plots,
        )

