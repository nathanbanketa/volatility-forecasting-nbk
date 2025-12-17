"""
Training script for volatility forecasting models.
Supports LSTM and GRU models.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import build_spy_pipeline
from src.models.lstm import LSTMVolatility
from src.models.gru import GRUVolatility


def train_model(
    model_type: str = "lstm",
    csv_path: str = None,
    model_save_path: str = None,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 64,
    num_epochs: int = 40,
    learning_rate: float = 1e-3,
    device: str = None,
):
    """
    Train a volatility forecasting model.
    
    Args:
        model_type: "lstm" or "gru"
        csv_path: Path to CSV file with returns data
        model_save_path: Path to save trained model
        split_date: Date to split train/test
        vol_window: Window size for realized volatility calculation
        seq_window: Sequence window size
        hidden_size: Hidden size for RNN
        num_layers: Number of RNN layers
        dropout: Dropout rate
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to use ("cuda" or "cpu")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    if csv_path is None:
        csv_path = project_root / "data" / "SPY_returns.csv"
    else:
        csv_path = Path(csv_path)
    
    if model_save_path is None:
        model_save_path = project_root / "models" / f"{model_type}_best.pt"
    else:
        model_save_path = Path(model_save_path)
    
    # Ensure models directory exists
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {csv_path}")
    train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq = build_spy_pipeline(
        csv_path=str(csv_path),
        split_date=split_date,
        vol_window=vol_window,
        seq_window=seq_window,
    )

    X_train_seq = np.array(X_train_seq)
    X_test_seq = np.array(X_test_seq)
    y_train_seq = np.array(y_train_seq).reshape(-1)
    y_test_seq = np.array(y_test_seq).reshape(-1)

    # Normalize features
    feature_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
    feature_std = X_train_seq.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train_seq_norm = (X_train_seq - feature_mean) / feature_std
    X_test_seq_norm = (X_test_seq - feature_mean) / feature_std

    X_train = torch.tensor(X_train_seq_norm, dtype=torch.float32)
    y_train = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test = torch.tensor(X_test_seq_norm, dtype=torch.float32)
    y_test = torch.tensor(y_test_seq, dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[-1]

    # Create model
    if model_type.lower() == "lstm":
        model = LSTMVolatility(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    elif model_type.lower() == "gru":
        model = GRUVolatility(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lstm' or 'gru'.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use scheduler for GRU (like in original train_gru_vol.py)
    if model_type.lower() == "gru":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    print(f"Training {model_type.upper()} model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"Train Loss: {np.mean(train_losses):.6f} | "
            f"Test RMSE: {rmse:.6f}"
        )
        
        if scheduler is not None:
            scheduler.step()

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test.to(device)).cpu().numpy()

    final_mse = mean_squared_error(y_test_seq, final_preds)
    final_rmse = np.sqrt(final_mse)
    print(f"Final {model_type.upper()} Test RMSE: {final_rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train volatility forecasting models")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"],
                        help="Model type: lstm or gru")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV file with returns data")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save trained model")
    parser.add_argument("--split-date", type=str, default="2020-01-01",
                        help="Date to split train/test")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Hidden size for RNN")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model,
        csv_path=args.csv_path,
        model_save_path=args.save_path,
        split_date=args.split_date,
        num_epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

