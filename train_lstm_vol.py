import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_pipeline import build_spy_pipeline
from src.models_rnn import LSTMVolatility


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = project_root / "SPY_returns.csv"

train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq = build_spy_pipeline(
    csv_path=str(csv_path),
    split_date="2020-01-01",
    vol_window=21,
    seq_window=20,
)

X_train_seq = np.array(X_train_seq)
X_test_seq = np.array(X_test_seq)
y_train_seq = np.array(y_train_seq).reshape(-1)
y_test_seq = np.array(y_test_seq).reshape(-1)

X_train = torch.tensor(X_train_seq, dtype=torch.float32)
y_train = torch.tensor(y_train_seq, dtype=torch.float32)
X_test = torch.tensor(X_test_seq, dtype=torch.float32)
y_test = torch.tensor(y_test_seq, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

input_size = X_train.shape[-1]

model = LSTMVolatility(
    input_size=input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 20

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

model.eval()
with torch.no_grad():
    final_preds = model(X_test.to(device)).cpu().numpy()

final_mse = mean_squared_error(y_test_seq, final_preds)
final_rmse = np.sqrt(final_mse)
print("Final LSTM Test RMSE:", final_rmse)
