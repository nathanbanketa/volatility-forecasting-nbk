import yfinance as yf
import pandas as pd
from datetime import datetime

def download_spy_returns():
    ticker = "SPY"
    start_date = "2010-01-01"

    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")

    spy = yf.Ticker(ticker)
    hist = spy.history(start=start_date, end=end_date)

    returns = hist['Close'].pct_change()

    df = pd.DataFrame({
        'Date': returns.index,
        'Returns': returns.values
    })

    df = df.dropna()

    df['Date'] = pd.to_datetime(df['Date']).dt.date

    df = df.reset_index(drop=True)

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

