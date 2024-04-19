import numpy
import pandas
import yfinance as yf


# Downloads df from yfinance and returns a dataframe with the col renamed.
# Calculates adj high and adj low
def download_df(ticker='AAPL', period='1y', interval='1d'):
    # period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    # interval: 1m, 2m, 5m, 15m, 30, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    df = yf.download(ticker, period=period, interval=interval)

    # Lower case col names
    df.columns = [x.lower() for x in df.columns]
    df.rename(columns={'adj close': 'adj_close'}, inplace=True)

    # Add adj high and low, which may be important later
    df['adj_high'] = df['adj_close'] * df['high'] / df['close']
    df['adj_low'] = df['adj_close'] * df['low'] / df['close']

    # NORMALIZATION STEP
    # df = df.log(df)
    
    return df
