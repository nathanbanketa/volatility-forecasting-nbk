"""
Download SPY (S&P 500 ETF) returns data from 2010 to present.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime


def download_spy_returns():
    """
    Download SPY historical data and calculate returns.
    
    Returns:
        pd.DataFrame: DataFrame with date and returns
    """
    # Define the ticker symbol
    ticker = "SPY"
    
    # Define start date (2010-01-01)
    start_date = "2010-01-01"
    
    # Get today's date
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    
    # Download the stock data
    spy = yf.Ticker(ticker)
    hist = spy.history(start=start_date, end=end_date)
    
    # Calculate returns (percentage change)
    returns = hist['Close'].pct_change()
    
    # Create a DataFrame with date and returns
    df = pd.DataFrame({
        'Date': returns.index,
        'Returns': returns.values
    })
    
    # Remove the first row (NaN due to pct_change)
    df = df.dropna()
    
    # Format dates as YYYY-MM-DD (remove time and timezone)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save to CSV
    output_file = "SPY_returns.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Data saved to {output_file}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


if __name__ == "__main__":
    df = download_spy_returns()
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())

