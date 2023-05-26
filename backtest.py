import pandas as pd
import datetime as dt
import time
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch.unitroot.cointegration import DynamicOLS
# from binance.client import Client
# from binance.exceptions import BinanceAPIException, BinanceOrderException
import requests
from io import StringIO
from pykalman import KalmanFilter
from pykalman import UnscentedKalmanFilter
import statsmodels.api as sm
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin

import yfinance as yf
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import statsmodels.tsa.stattools as ts
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.stattools import coint
# import missingno
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator

import heapq
import operator
from pandas_datareader import data as web

import warnings
warnings.simplefilter('ignore')


class Backtest:
    @staticmethod
    def backtest_signal_sigma(df, params):
        roll_window = params[0]
        pos = 'nul'
        ptf = 100
        fees = 0.001
        nb_eth = 0
        nb_btc = 0
        val_trade = 0
        pnl = []
        df['pos_btc'] = 0
        df['pos_eth'] = 0
        df['seuil_signal'] = df['misp_ptf'].rolling(roll_window).std()
        df['pnl_strat'] = 0
        for i in range(roll_window, df.shape[0]):
            if pos == 'nul':
                if df['misp_ptf'].iloc[i] < -df['seuil_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] >= - \
                df['seuil_signal'].iloc[i - 1]:
                    pos = 'short_eth'
                    df.loc[i, 'pos_eth'] = 1
                    df.loc[i, 'pos_btc'] = -1
                    pos_size = ptf/(df['beta'].iloc[i]*df['eth_close'].iloc[i]+df['btc_close'].iloc[i])
                    nb_eth = pos_size*df['beta'].iloc[i]
                    nb_btc = pos_size
                    val_trade = nb_eth * df['eth_close'].iloc[i] - 1.0 * nb_btc * df['btc_close'].iloc[i]
                    ptf = 0
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
                elif df['misp_ptf'].iloc[i] > df['seuil_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] <= \
                        df['seuil_signal'].iloc[i - 1]:
                    pos = 'short_btc'
                    df.loc[i, 'pos_eth'] = -1
                    df.loc[i, 'pos_btc'] = 1
                    pos_size = ptf/(df['beta'].iloc[i]*df['eth_close'].iloc[i]+df['btc_close'].iloc[i])
                    nb_eth = pos_size*df['beta'].iloc[i]
                    nb_btc = pos_size
                    val_trade = -1.0 * nb_eth * df['eth_close'].iloc[i] + nb_btc * df['btc_close'].iloc[i]
                    ptf = 0
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']

                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
            elif pos == 'short_eth':
                if df['misp_ptf'].iloc[i] > 0 and df['misp_ptf'].iloc[i - 1] <= 0:
                    pos = 'nul'
                    # long eth qu'on a short pr 1000e, short btc qu'on a long pour 1000e
                    df.loc[i, 'pos_eth'] = -1
                    df.loc[i, 'pos_btc'] = 1
                    pnl.append(-1.0 * nb_eth * df['eth_close'].iloc[i] + nb_btc * df['btc_close'].iloc[i] + val_trade - (ptf*fees*2))
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat'] + pnl[-1]
                    ptf = 100
                    val_trade = 0
                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
            else:
                if df['misp_ptf'].iloc[i] < 0 and df['misp_ptf'].iloc[i - 1] >= 0:
                    pos = 'nul'
                    # short eth qu'on a long pr 1000e, long btc qu'on a short pour 1000e
                    df.loc[i + 1, 'pos_eth'] = 1
                    df.loc[i + 1, 'pos_btc'] = -1
                    pnl.append(nb_eth * df['eth_close'].iloc[i] - 1.0 * nb_btc * df['btc_close'].iloc[i] + val_trade - (ptf*fees*2))
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat'] + pnl[-1]
                    ptf = 100
                    val_trade = 0
                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']

        if len(pnl) != 0:
            final_pnl = np.cumsum(pnl)[-1]
        else:
            final_pnl = 0

        return df, final_pnl, pnl

    @staticmethod
    def simple_backtest(df, params):
        coeff_const = params[0]
        coeff_std = params[1]
        roll_window = params[2]
        pos = 'nul'
        ptf = 100
        fees = 0.001
        nb_eth = 0
        nb_btc = 0
        val_trade = 0
        pnl = []
        df['pos_btc'] = 0
        df['pos_eth'] = 0
        df['roll_vol'] = df['misp_ptf'].rolling(roll_window).std()
        df['seuil_signal'] = coeff_const + coeff_std * df['roll_vol']
        df['pnl_strat'] = 0
        # df['seuil_signal'] = coeff_const
        for i in range(roll_window, df.shape[0]):
            if pos == 'nul':
                if df['misp_ptf'].iloc[i] < -df['seuil_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] >= - \
                df['seuil_signal'].iloc[i - 1]:
                    pos = 'short_eth'
                    df.loc[i, 'pos_eth'] = 1
                    df.loc[i, 'pos_btc'] = -1
                    pos_size = ptf/(df['beta'].iloc[i]*df['eth_close'].iloc[i]+df['btc_close'].iloc[i])
                    nb_eth = pos_size*df['beta'].iloc[i]
                    nb_btc = pos_size
                    val_trade = nb_eth * df['eth_close'].iloc[i] - 1.0 * nb_btc * df['btc_close'].iloc[i]
                    ptf = 0
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
                elif df['misp_ptf'].iloc[i] > df['seuil_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] <= \
                        df['seuil_signal'].iloc[i - 1]:
                    pos = 'short_btc'
                    df.loc[i, 'pos_eth'] = -1
                    df.loc[i, 'pos_btc'] = 1
                    pos_size = ptf/(df['beta'].iloc[i]*df['eth_close'].iloc[i]+df['btc_close'].iloc[i])
                    nb_eth = pos_size*df['beta'].iloc[i]
                    nb_btc = pos_size
                    val_trade = -1.0 * nb_eth * df['eth_close'].iloc[i] + nb_btc * df['btc_close'].iloc[i]
                    ptf = 0
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']

                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
            elif pos == 'short_eth':
                if df['misp_ptf'].iloc[i] > 0 and df['misp_ptf'].iloc[i - 1] <= 0:
                    pos = 'nul'
                    # long eth qu'on a short pr 1000e, short btc qu'on a long pour 1000e
                    df.loc[i, 'pos_eth'] = -1
                    df.loc[i, 'pos_btc'] = 1
                    pnl.append(-1.0 * nb_eth * df['eth_close'].iloc[i] + nb_btc * df['btc_close'].iloc[i] + val_trade - (ptf*fees*2))
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat'] + pnl[-1]
                    ptf = 100
                    val_trade = 0
                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
            else:
                if df['misp_ptf'].iloc[i] < 0 and df['misp_ptf'].iloc[i - 1] >= 0:
                    pos = 'nul'
                    # short eth qu'on a long pr 1000e, long btc qu'on a short pour 1000e
                    df.loc[i + 1, 'pos_eth'] = 1
                    df.loc[i + 1, 'pos_btc'] = -1
                    pnl.append(nb_eth * df['eth_close'].iloc[i] - 1.0 * nb_btc * df['btc_close'].iloc[i] + val_trade - (ptf*fees*2))
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat'] + pnl[-1]
                    ptf = 100
                    val_trade = 0
                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']

        if len(pnl) != 0:
            final_pnl = np.cumsum(pnl)[-1]
        else:
            final_pnl = 0

        return df, final_pnl, pnl
