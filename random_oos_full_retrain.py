import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_pipeline import _load_returns_csv, _fetch_vix_series
from src.models_garch import garch_model
from src.models_rnn import LSTMVolatility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = project_root / "SPY_returns.csv"


df = _load_returns_csv(str(csv_path))

vol_window = 21
seq_window = 20
use_vix = True

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

feature_cols = ["ret", "abs_ret", "ret_sq", "realized_vol_lag1"]
if use_vix:
    feature_cols.append("vix")
target_col = "realized_vol"


X_list = []
y_list = []
idx_list = []

data = df[feature_cols]
target = df[target_col]

for i in range(seq_window, len(df)):
    window = data.iloc[i - seq_window:i].values     
    y_val = target.iloc[i]                 
    X_list.append(window)
    y_list.append(y_val)
    idx_list.append(df.index[i])                

X_all_seq = np.array(X_list)  
y_all_seq = np.array(y_list)     
seq_index = pd.Index(idx_list)  

print("Total sequences:", X_all_seq.shape[0])

res, garch_vol_full = garch_model(df["ret"], p=1, q=1)
garch_vol_full.index = df.index
garch_seq_all = garch_vol_full.loc[seq_index].values

N = len(X_all_seq)
N_SAMPLE = 100

rng = np.random.default_rng(42) #keep seed consistent
if N_SAMPLE > N:
    N_SAMPLE = N

test_idx = rng.choice(N, size=N_SAMPLE, replace=False)
train_idx = np.setdiff1d(np.arange(N), test_idx)

X_train_seq = X_all_seq[train_idx]
y_train_seq = y_all_seq[train_idx]
X_test_seq = X_all_seq[test_idx]
y_test_seq = y_all_seq[test_idx]
garch_test = garch_seq_all[test_idx]
index_test = seq_index[test_idx]

print(f"Train sequences: {len(train_idx)}, Test sequences (random OoS): {len(test_idx)}")


feature_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
feature_std = X_train_seq.std(axis=(0, 1), keepdims=True) + 1e-8

X_train_norm = (X_train_seq - feature_mean) / feature_std
X_test_norm = (X_test_seq - feature_mean) / feature_std


X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
y_test_t = torch.tensor(y_test_seq, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

input_size = X_train_t.shape[-1]

model = LSTMVolatility(
    input_size=input_size,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 40

print("=== Training LSTM on non-test (random OoS) sequences ===")
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
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"Train Loss: {np.mean(train_losses):.6f} | "
        f"Random-OoS Test RMSE: {rmse:.6f}"
    )


model.eval()
with torch.no_grad():
    lstm_pred = model(X_test_t.to(device)).cpu().numpy()

df_sample = pd.DataFrame(
    {
        "realized_vol": y_test_seq,
        "garch_vol": garch_test,
        "lstm_vol": lstm_pred.reshape(-1),
    },
    index=index_test,
).sort_index()

garch_rmse_sample = np.sqrt(
    mean_squared_error(df_sample["realized_vol"], df_sample["garch_vol"])
)
lstm_rmse_sample = np.sqrt(
    mean_squared_error(df_sample["realized_vol"], df_sample["lstm_vol"])
)

print("\n=== Random Out-of-Sample over FULL history (with retraining) ===")
print(f"Subset size: {len(df_sample)}")
print("GARCH RMSE (random OoS):", garch_rmse_sample)
print("LSTM RMSE (random OoS): ", lstm_rmse_sample)


plt.figure(figsize=(12, 5))
plt.plot(df_sample.index, df_sample["realized_vol"], label="Realized Vol", alpha=0.7)
plt.plot(df_sample.index, df_sample["garch_vol"], label="GARCH Vol", alpha=0.7)
plt.plot(df_sample.index, df_sample["lstm_vol"], label="LSTM Vol", alpha=0.7)
plt.title("Volatility Forecasts vs Realized Volatility (Random OoS Subset, Full History)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.scatter(df_sample["realized_vol"], df_sample["garch_vol"],
            alpha=0.4, label="GARCH")
plt.scatter(df_sample["realized_vol"], df_sample["lstm_vol"],
            alpha=0.4, label="LSTM")
plt.plot(
    [df_sample["realized_vol"].min(), df_sample["realized_vol"].max()],
    [df_sample["realized_vol"].min(), df_sample["realized_vol"].max()],
    'k--', label="Perfect Fit"
)
plt.title("Predicted vs Actual (Random OoS Subset, Full History)")
plt.xlabel("Realized Vol")
plt.ylabel("Predicted Vol")
plt.legend()
plt.tight_layout()
plt.show()

garch_errors = df_sample["realized_vol"] - df_sample["garch_vol"]
lstm_errors = df_sample["realized_vol"] - df_sample["lstm_vol"]

plt.figure(figsize=(12, 5))
plt.hist(garch_errors, bins=30, alpha=0.5, label="GARCH Errors")
plt.hist(lstm_errors, bins=30, alpha=0.5, label="LSTM Errors")
plt.title("Residual Distribution Comparison (Random OoS Subset, Full History)")
plt.xlabel("Error (Realized - Predicted)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
