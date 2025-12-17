"""
Run GARCH baseline model.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data import build_spy_pipeline
from src.models.garch import garch_model


def run_baseline(
    csv_path: str = None,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20,
    p: int = 1,
    q: int = 1,
):
    """
    Run GARCH baseline model and print test RMSE.
    
    Args:
        csv_path: Path to CSV file with returns data
        split_date: Date to split train/test
        vol_window: Window size for realized volatility calculation
        seq_window: Sequence window size
        p: GARCH p parameter
        q: GARCH q parameter
    """
    if csv_path is None:
        csv_path = project_root / "data" / "SPY_returns.csv"
    else:
        csv_path = Path(csv_path)
    
    train_df, test_df, X_train_seq, y_train_seq, X_test_seq, y_test_seq = build_spy_pipeline(
        csv_path=str(csv_path),
        split_date=split_date,
        vol_window=vol_window,
        seq_window=seq_window,
    )

    returns = pd.concat([train_df["ret"], test_df["ret"]])

    res, garch_vol_full = garch_model(
        returns=returns,
        p=p,
        q=q,
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
    return rmse


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GARCH baseline model")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV file with returns data")
    parser.add_argument("--split-date", type=str, default="2020-01-01",
                        help="Date to split train/test")
    parser.add_argument("--p", type=int, default=1,
                        help="GARCH p parameter")
    parser.add_argument("--q", type=int, default=1,
                        help="GARCH q parameter")
    
    args = parser.parse_args()
    
    run_baseline(
        csv_path=args.csv_path,
        split_date=args.split_date,
        p=args.p,
        q=args.q,
    )

