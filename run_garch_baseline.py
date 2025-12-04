import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_pipeline import build_spy_pipeline
from src.models_garch import garch_model

csv_path = project_root / "SPY_returns.csv"

train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq = build_spy_pipeline(
    csv_path=str(csv_path),
    split_date="2020-01-01",
    vol_window=21,
    seq_window=20,
)

returns = pd.concat([train_df["ret"], test_df["ret"]])

res, garch_vol_full = garch_model(
    returns=returns,
    p=1,
    q=1,
)

full_index = returns.index
garch_vol_full.index = full_index

garch_vol_test = garch_vol_full.loc[test_df.index]
realized_vol_test = test_df["realized_vol"]

mask = realized_vol_test.notna() & garch_vol_test.notna()
y_true = realized_vol_test[mask].values
y_pred = garch_vol_test[mask].values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("GARCH Test RMSE:", rmse)
