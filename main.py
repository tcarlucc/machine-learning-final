"""
Main file for accumulating examples of breakout trades.
Goal is to use example trades to train a model on these metalabeled trades.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from trendline_break_dataset import trendline_breakout_dataset
import matplotlib.pyplot as plt

DEFAULT_LOOKBACK = 168


def rename_cols(data: pd.DataFrame):
    """
    Change names of yfinance data to fit the trendline_breakout_dataset functions

    :param data: yfinance pandas dataframe to be modified
    :return: modified dataframe
    """
    data.rename(columns={'Open': 'open'}, inplace=True)
    data.rename(columns={'Close': 'close'}, inplace=True)
    data.drop('Adj Close', axis=1)
    data.rename(columns={'High': 'high'}, inplace=True)
    data.rename(columns={'Low': 'low'}, inplace=True)
    data.rename(columns={'Volume': 'volume'}, inplace=True)
    return data


if __name__ == "__main__":
    # Download stock data for a variety of tickers, blue chips for now
    # Period params: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # Interval params: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    aapl = rename_cols(yf.download('AAPL', period='1y', interval='1h'))
    msft = rename_cols(yf.download('MSFT', period='1y', interval='1h'))

    # Run trendline_breakout_dataset and accumulate examples
    trades, data_x, data_y = trendline_breakout_dataset(aapl, DEFAULT_LOOKBACK)
    print(trades)

    # Plotting to get familiar with trades data structure
    fig, ax = plt.subplots()