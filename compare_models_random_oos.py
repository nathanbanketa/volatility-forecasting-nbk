import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_pipeline import build_spy_pipeline
from src.models_garch import garch_model
from src.models_rnn import LSTMVolatility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = project_root / "SPY_returns.csv"

train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq = build_spy_pipeline(
    csv_path=str(csv_path),
    split_date="2020-01-01",
    vol_window=21,
    seq_window=20,
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

state_path = project_root / "lstm_v2_best.pt"
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

print("Lengths:",
      "seq_index", len(seq_index),
      "realized_seq", realized_seq.shape[0],
      "garch_seq", garch_seq.shape[0],
      "lstm_seq", lstm_seq.shape[0])

df_compare = pd.DataFrame(
    {
        "realized_vol": realized_seq,
        "garch_vol": garch_seq,
        "lstm_vol": lstm_seq,
    },
    index=seq_index,
)

N_SAMPLE = 100

rng = np.random.default_rng(42)
if N_SAMPLE > len(df_compare):
    N_SAMPLE = len(df_compare)

sample_indices = rng.choice(len(df_compare), size=N_SAMPLE, replace=False)
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
