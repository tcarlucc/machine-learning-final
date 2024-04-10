import numpy
import pandas
import yfinance as yf


# Downloads df from yfinance and returns a dataframe with the col renamed.
# Calculates adj high and adj low
def download_df(ticker, period, interval='1d'):
    # period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    # interval: 1m, 2m, 5m, 15m, 30, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

    df = yf.download(ticker, period=period, interval=interval)
    # df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    # # Add adj high and low, which may be important later
    # df['Adj_High'] = df['Adj_Close'] * df['High'] / df['Close']
    # df['Adj_Low'] = df['Adj_Close'] * df['Low'] / df['Close']

    # Lower case col names
    df.columns = [x.lower() for x in df.columns]

    # NORMALIZATION STEP
    # df = df.log(df)
    
    return df


