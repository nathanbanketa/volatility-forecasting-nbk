"""
Paper Trading Experiment using LSTM Volatility Predictions

This script uses the trained LSTM model to make volatility predictions
and generates trading signals for VIXY (long volatility) or SVIX (short volatility)
based on whether predicted volatility is higher or lower than current realized volatility.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.lstm import LSTMVolatility
from src.data import _fetch_vix_series


def prepare_features_from_live_data(
    spy_data: pd.DataFrame,
    vix_series: pd.Series,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Prepare features matching the training pipeline exactly.
    
    Args:
        spy_data: DataFrame with SPY price data (must have 'Close' column)
        vix_series: Series with VIX values
        vol_window: Window size for realized volatility calculation
    
    Returns:
        DataFrame with all features ready for model input
    """
    df = pd.DataFrame(index=spy_data.index)
    
    # Calculate returns (log returns to match training if needed, or pct_change)
    # Training uses pct_change based on data_pipeline
    df["ret"] = spy_data["Close"].pct_change()
    
    # Feature engineering (matching src/data.py exactly)
    df["realized_vol"] = df["ret"].rolling(vol_window).std()
    df["abs_ret"] = df["ret"].abs()
    df["ret_sq"] = df["ret"] ** 2
    df["realized_vol_lag1"] = df["realized_vol"].shift(1)
    
    # Add VIX (lowercase 'vix' to match training)
    vix_aligned = vix_series.reindex(df.index)
    df["vix"] = vix_aligned.ffill()
    
    # Drop rows with NaN
    df = df.dropna()
    
    return df


def get_scaler_statistics(
    df: pd.DataFrame,
    split_date: str = "2020-01-01",
    seq_window: int = 20,
) -> tuple:
    """
    Recreate scaler statistics from training data.
    
    Args:
        df: DataFrame with features
        split_date: Date where training data ends
        seq_window: Sequence window size
    
    Returns:
        Tuple of (mean, std) for feature normalization
    """
    split_ts = pd.to_datetime(split_date)
    train_df = df[df.index < split_ts].copy()
    
    if len(train_df) < seq_window:
        raise ValueError(f"Not enough training data. Need at least {seq_window} rows.")
    
    feature_cols = ["ret", "abs_ret", "ret_sq", "realized_vol_lag1", "vix"]
    
    # Get all sequences from training data to compute proper statistics
    X_list = []
    for i in range(seq_window, len(train_df)):
        window = train_df[feature_cols].iloc[i - seq_window:i].values
        X_list.append(window)
    
    X_train_seq = np.array(X_list)
    
    # Compute mean and std across all sequences (matching training)
    feature_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
    feature_std = X_train_seq.std(axis=(0, 1), keepdims=True) + 1e-8
    
    return feature_mean, feature_std


def predict_volatility(
    model: nn.Module,
    df: pd.DataFrame,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    seq_window: int = 20,
    device: str = "cpu",
) -> float:
    """
    Make volatility prediction using the LSTM model.
    
    Args:
        model: Trained LSTM model
        df: DataFrame with features
        feature_mean: Mean for normalization
        feature_std: Std for normalization
        seq_window: Sequence window size
        device: Device to run inference on
    
    Returns:
        Predicted volatility value
    """
    feature_cols = ["ret", "abs_ret", "ret_sq", "realized_vol_lag1", "vix"]
    
    if len(df) < seq_window:
        raise ValueError(f"Need at least {seq_window} rows of data for prediction.")
    
    # Get last sequence
    last_sequence = df[feature_cols].tail(seq_window).values
    
    # Normalize
    last_sequence_norm = (last_sequence - feature_mean) / feature_std
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(
        last_sequence_norm,
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).cpu().item()
    
    return prediction


def get_etf_prices(tickers: list) -> dict:
    """
    Fetch current prices for ETFs.
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Dictionary mapping ticker to price
    """
    prices = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="1d")
            if len(hist) > 0:
                prices[ticker] = hist["Close"].iloc[-1]
            else:
                prices[ticker] = None
        except Exception as e:
            print(f"Warning: Could not fetch price for {ticker}: {e}")
            prices[ticker] = None
    return prices


def run_paper_trading_experiment(
    model_path: str = None,
    csv_path: str = None,
    split_date: str = "2020-01-01",
    vol_window: int = 21,
    seq_window: int = 20,
    long_vol_etf: str = "VIXY",
    short_vol_etf: str = "SVIX",
    trade_amount: float = 1000.0,
    device: str = None,
    use_live_data: bool = True,
):
    """
    Run paper trading experiment using LSTM volatility predictions.
    
    Args:
        model_path: Path to trained model file
        csv_path: Path to CSV with historical data (if not using live data)
        split_date: Date where training data ends
        vol_window: Window size for realized volatility
        seq_window: Sequence window size
        long_vol_etf: Ticker for long volatility ETF
        short_vol_etf: Ticker for short volatility ETF
        trade_amount: Dollar amount to "trade"
        device: Device to run model on
        use_live_data: If True, fetch live data from yfinance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    if model_path is None:
        model_path = project_root / "models" / "lstm_v2_best.pt"
    else:
        model_path = Path(model_path)
    
    print("=" * 60)
    print("PAPER TRADING EXPERIMENT - LSTM Volatility Prediction")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load or fetch data
    if use_live_data:
        print("Fetching live market data...")
        spy_ticker = yf.Ticker("SPY")
        spy_data = spy_ticker.history(start="2010-01-01", progress=False)
        
        if "Close" not in spy_data.columns:
            # Handle multi-index if present
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data = spy_data["Close"]
            else:
                spy_data = spy_data.iloc[:, 0]  # Take first column
        
        # Get VIX data
        start_date = spy_data.index.min()
        end_date = spy_data.index.max()
        vix_series = _fetch_vix_series(start_date, end_date)
        
        print(f"Loaded {len(spy_data)} days of SPY data")
        print(f"Date range: {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
    else:
        if csv_path is None:
            csv_path = project_root / "data" / "SPY_returns.csv"
        else:
            csv_path = Path(csv_path)
        
        print(f"Loading data from {csv_path}...")
        from src.data import _load_returns_csv, build_spy_pipeline
        
        # For CSV, we need to reconstruct the full pipeline
        # This is more complex, so we'll use the build_spy_pipeline approach
        train_df, test_df, _, _, _, _ = build_spy_pipeline(
            csv_path=str(csv_path),
            split_date=split_date,
            vol_window=vol_window,
            seq_window=seq_window,
        )
        
        # Combine train and test for full history
        spy_data = pd.concat([train_df, test_df])
        # Reconstruct Close prices from returns (approximate)
        spy_data["Close"] = (1 + spy_data["ret"]).cumprod() * 100  # Normalize
        
        start_date = spy_data.index.min()
        end_date = spy_data.index.max()
        vix_series = _fetch_vix_series(start_date, end_date)
    
    # Prepare features
    print("\nPreparing features...")
    df = prepare_features_from_live_data(spy_data, vix_series, vol_window=vol_window)
    print(f"Features prepared. Total rows: {len(df)}")
    
    # Get scaler statistics from training data
    print("\nComputing normalization statistics from training data...")
    feature_mean, feature_std = get_scaler_statistics(df, split_date, seq_window)
    print("Scaler statistics computed.")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Please train a model first using scripts/train.py"
        )
    
    input_size = 5  # ["ret", "abs_ret", "ret_sq", "realized_vol_lag1", "vix"]
    model = LSTMVolatility(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully.")
    
    # Make prediction
    print("\nMaking volatility prediction...")
    predicted_vol = predict_volatility(
        model, df, feature_mean, feature_std, seq_window, device
    )
    
    # Get current realized volatility
    current_realized_vol = df["realized_vol"].iloc[-1]
    current_date = df.index[-1]
    
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Date: {current_date.date()}")
    print(f"Current Realized Volatility: {current_realized_vol:.6f}")
    print(f"LSTM Predicted Volatility:   {predicted_vol:.6f}")
    print(f"Difference:                  {predicted_vol - current_realized_vol:.6f}")
    print()
    
    # Get ETF prices
    print("Fetching ETF prices...")
    etf_prices = get_etf_prices([long_vol_etf, short_vol_etf])
    vixy_price = etf_prices.get(long_vol_etf)
    svix_price = etf_prices.get(short_vol_etf)
    
    # Trading decision
    print(f"\n{'='*60}")
    print("TRADING DECISION")
    print(f"{'='*60}")
    
    volatility_expanding = predicted_vol > current_realized_vol
    
    if volatility_expanding:
        print("📈 SIGNAL: VOLATILITY EXPANDING")
        print("   → Model predicts volatility will increase")
        print("   → Strategy: Long Volatility (Buy VIXY)")
        print()
        print(f"Action: Buy ${trade_amount:.2f} of {long_vol_etf}")
        if vixy_price and vixy_price > 0:
            shares = trade_amount / vixy_price
            print(f"Current {long_vol_etf} Price: ${vixy_price:.2f}")
            print(f"Shares to Buy: {shares:.4f}")
            print(f"Total Cost: ${shares * vixy_price:.2f}")
        else:
            print(f"⚠️  Could not fetch {long_vol_etf} price")
    else:
        print("📉 SIGNAL: VOLATILITY CONTRACTING")
        print("   → Model predicts volatility will decrease")
        print("   → Strategy: Short Volatility (Buy SVIX)")
        print()
        print(f"Action: Buy ${trade_amount:.2f} of {short_vol_etf}")
        if svix_price and svix_price > 0:
            shares = trade_amount / svix_price
            print(f"Current {short_vol_etf} Price: ${svix_price:.2f}")
            print(f"Shares to Buy: {shares:.4f}")
            print(f"Total Cost: ${shares * svix_price:.2f}")
        else:
            print(f"⚠️  Could not fetch {short_vol_etf} price")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    return {
        "date": current_date,
        "current_realized_vol": current_realized_vol,
        "predicted_vol": predicted_vol,
        "signal": "LONG_VOL" if volatility_expanding else "SHORT_VOL",
        "etf": long_vol_etf if volatility_expanding else short_vol_etf,
        "trade_amount": trade_amount,
        "etf_price": vixy_price if volatility_expanding else svix_price,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper trading experiment using LSTM volatility predictions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model file (default: models/lstm_v2_best.pt)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to CSV with historical data (if not using live data)",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2020-01-01",
        help="Date where training data ends",
    )
    parser.add_argument(
        "--long-vol-etf",
        type=str,
        default="VIXY",
        help="Ticker for long volatility ETF",
    )
    parser.add_argument(
        "--short-vol-etf",
        type=str,
        default="SVIX",
        help="Ticker for short volatility ETF",
    )
    parser.add_argument(
        "--trade-amount",
        type=float,
        default=1000.0,
        help="Dollar amount to trade",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        help="Use CSV file instead of live data",
    )
    
    args = parser.parse_args()
    
    run_paper_trading_experiment(
        model_path=args.model_path,
        csv_path=args.csv_path,
        split_date=args.split_date,
        long_vol_etf=args.long_vol_etf,
        short_vol_etf=args.short_vol_etf,
        trade_amount=args.trade_amount,
        device=args.device,
        use_live_data=not args.use_csv,
    )

