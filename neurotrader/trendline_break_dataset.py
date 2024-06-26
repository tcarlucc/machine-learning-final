import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from neurotrader.trendline_automation import fit_trendlines_single, fit_upper_trendline
import mplfinance as mpf
import yfinance as yf


def atr_calc(p_data: pd.DataFrame, n: int):
    """
    Average True Range calculator. Original file used pandas_ta's implementation of this but seems to not be working.
    Indicates degree of price volatility, not necessarily price trends.
    Formula can be found here: https://en.wikipedia.org/wiki/Average_true_range
    Help from: https://stackoverflow.com/questions/40256338/calculating-average-true-range-atr-on-ohlc-data-with-python

    :param p_data: Pandas dataframe of stock data
    :param n: Number of datapoints to average over
    :return: Series of average true ranges for the data
    """
    f_data = p_data.copy()
    high = np.log(f_data['High'])
    low = np.log(f_data['Low'])
    close = np.log(f_data['Close'])
    f_data['tr0'] = abs(high - low)
    f_data['tr1'] = abs(high - close.shift())
    f_data['tr2'] = abs(low - close.shift())
    tr = f_data[['tr0', 'tr1', 'tr2']].max(axis=1)
    result = tr.ewm(alpha=1/n, adjust=False).mean()  # J. Welles Wilder's Exponential Moving Average
    return result


def trendline_breakout_dataset(
        ohlcv: pd.DataFrame, lookback: int, 
        hold_period: int = 12, tp_mult: float = 3.0, sl_mult: float = 3.0,
        atr_lookback: int = 168
):
    assert(atr_lookback >= lookback)

    close = np.log(ohlcv['Close'].to_numpy())

    # ATR for normalizing, setting stop loss take profit
    atr = atr_calc(ohlcv, atr_lookback)
    atr_arr = atr.to_numpy()

    # Normalized volume
    vol_arr = (ohlcv['Volume'] / ohlcv['Volume'].rolling(atr_lookback).median()).to_numpy()
    adx = ta.adx(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], lookback)
    adx_arr = adx['ADX_' + str(lookback)].to_numpy()

    trades = pd.DataFrame()
    trade_i = 0

    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
    for i in range(atr_lookback, len(ohlcv)):
        # NOTE window does NOT include the current candle
        window = close[i - lookback: i]

        s_coefs, r_coefs = fit_trendlines_single(window)

        # Find current value of line
        r_val = r_coefs[1] + lookback * r_coefs[0]
        s_val = s_coefs[1] + lookback * s_coefs[0]

        # Entry
        if not in_trade and close[i] > r_val or close[i] < s_val:

            tp_price = close[i] + atr_arr[i] * tp_mult
            sl_price = close[i] - atr_arr[i] * sl_mult
            hp_i = i + hold_period
            in_trade = True

            trades.loc[trade_i, 'entry_i'] = i
            trades.loc[trade_i, 'entry_p'] = close[i]
            trades.loc[trade_i, 'atr'] = atr_arr[i]
            trades.loc[trade_i, 'sl'] = sl_price
            trades.loc[trade_i, 'tp'] = tp_price
            trades.loc[trade_i, 'hp_i'] = i + hold_period

            if close[i] > r_val:
                trades.loc[trade_i, 'slope'] = r_coefs[0]
                trades.loc[trade_i, 'intercept'] = r_coefs[1]

                # Trendline features
                # Resist slope
                trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i]
                trades.loc[trade_i, 'support_s'] = s_coefs[0] / atr_arr[i]

                # Resist error
                line_vals = (r_coefs[1] + np.arange(lookback) * r_coefs[0])
                err = np.sum(line_vals - window) / lookback
                err /= atr_arr[i]
                trades.loc[trade_i, 'tl_err'] = err

                # Max distance from resist
                diff = line_vals - window
                trades.loc[trade_i, 'max_dist'] = diff.max() / atr_arr[i]

                # Resist/Support break classifier
                trades.loc[trade_i, 'class'] = 1

            else:
                trades.loc[trade_i, 'slope'] = s_coefs[0]
                trades.loc[trade_i, 'intercept'] = s_coefs[1]

                # Resist & Support slope
                trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i]
                trades.loc[trade_i, 'support_s'] = s_coefs[0] / atr_arr[i]

                # Support error
                line_vals = s_coefs[1] + np.arange(lookback) * s_coefs[0]
                err = np.sum(line_vals - window) / lookback
                err /= atr_arr[i]
                trades.loc[trade_i, 'tl_err'] = err

                # Max distance from support
                diff = line_vals - window
                trades.loc[trade_i, 'max_dist'] = diff.min() / atr_arr[i]

                # Resist/Support break classifier
                trades.loc[trade_i, 'class'] = 0

            # Volume on breakout
            trades.loc[trade_i, 'vol'] = vol_arr[i]

            # ADX
            # trades.loc[trade_i, 'adx'] = adx_arr[i]

        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = close[i]
                
                in_trade = False
                trade_i += 1

    trades['return'] = trades['exit_p'] - trades['entry_p']
    trades.dropna(inplace=True)
    
    # Features
    data_x = trades[['resist_s', 'support_s', 'tl_err', 'vol', 'max_dist', 'class']]
    # Label
    data_y = pd.Series(0, index=trades.index)
    data_y.loc[trades['return'] > 0] = 1

    return trades, data_x, data_y


if __name__ == '__main__':
    """data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()"""

    data = yf.download('AAPL', period="1y", interval='1h')

    print(data['High'])
    # print(ta.atr(np.log(data['High']), np.log(data['low']), np.log(data['Close'])), 72)
    # print(np.log(data['High']), np.log(data['low']), np.log(data['Close']))

    trades, data_x, data_y = trendline_breakout_dataset(data, 168)

    # Drop any incomplete trades
    trades = trades.dropna()

    # Look at trades without any ML filter. 
    signal = np.zeros(len(data))
    for i in range(len(trades)):
        trade = trades.iloc[i]
        signal[int(trade['entry_i']):int(trade['exit_i'])] = 1.

    data['r'] = np.log(data['Close']).diff().shift(-1)
    data['sig'] = signal
    returns = data['r'] * data['sig']
    
    print("Profit Factor", returns[returns > 0].sum() / returns[returns < 0].abs().sum())
    print("Win Rate", len(trades[trades['return'] > 0]) / len(trades))
    print("Average Trade", trades['return'].mean())

    plt.style.use('dark_background')
    returns.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    #plt.show()





