import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf


def find_pips(data: np.array, n_pips: int, dist_measure: int):
    # dist_measure
    # 1 = Euclidean Distance
    # 2 = Perpindicular Distance
    # 3 = Vertical Distance
    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]]  # Price

    for curr_point in range(2, n_pips):

        md = 0.0  # Max distance
        md_i = -1  # Max distance index
        insert_index = -1

        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope;

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):

                d = 0.0  # Distance
                if dist_measure == 1:  # Euclidean distance
                    d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                elif dist_measure == 2:  # Perpindicular distance
                    d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                else:  # Vertical distance
                    d = abs((slope * i + intercept) - data[i])

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y


if __name__ == "__main__":
    # period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    # interval: 1m, 2m, 5m, 15m, 30, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

    PIP = 10

    data = yf.download('MSFT', period="1y", interval='1d')
    data.rename(columns={'Date': 'date'}, inplace=True)
    data.rename(columns={'Open': 'open'}, inplace=True)
    data.rename(columns={'Close': 'close'}, inplace=True)
    data.rename(columns={'High': 'high'}, inplace=True)
    data.rename(columns={'Low': 'low'}, inplace=True)
    data.rename(columns={'Volume': 'volume'}, inplace=True)

    i = 1198
    x = data['close'].to_numpy()
    print(data)

    pips_x, pips_y = find_pips(x, PIP, 1)

    pd.Series(x).plot()
    for i in range(PIP):
        plt.plot(pips_x[i], pips_y[i], marker='o', color='red')

    plt.show()
