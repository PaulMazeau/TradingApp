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


class GetSpread:
    def __init__(self, df, tick1, tick2):
        self.df = df
        self.tick1 = tick1
        self.tick2 = tick2

    def kalman_filter(self):
        x = self.df[self.tick1][:10]
        y = self.df[self.tick2][:10]
        delta = 1e-3
        # How much random walk wiggles
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
        # y is 1-dimensional, (alpha, beta) is 2-dimensional
        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
        initial_state_mean=[0,0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2,
        transition_covariance=trans_cov)
        # Use the observations y to get running estimates and errors for the state parameters
        state_means, state_covs = kf.filter(y.values)
        res_param = pd.DataFrame(state_means,
                                  index=range(len(x)),
                                  columns=['beta', 'alpha'])
        return res_param, kf, state_means, state_covs

    def udpate_kalman_filter(self, kf, history_state_means, history_state_covs):
        n = self.df.shape[0]
        n_dim_state = 2
        new_state_means = np.zeros((n, n_dim_state))
        new_state_covs = np.zeros((n, n_dim_state, n_dim_state))
        for i in range(self.df.shape[0]):
            obs_mat = np.asarray([[self.df[self.tick2].iloc[i], 1]])
            if i == 0:
                new_state_means[i], new_state_covs[i] = kf.filter_update(history_state_means[-1],
                                                                                 history_state_covs[-1],
                                                                                 observation=self.df[self.tick1].iloc[i],
                                                                                 observation_matrix=obs_mat)
            else:
                new_state_means[i], new_state_covs[i] = kf.filter_update(new_state_means[-1],
                                                                                 new_state_covs[-1],
                                                                                 observation=self.df[self.tick1].iloc[i],
                                                                                 observation_matrix=obs_mat)

        means = pd.DataFrame(new_state_means,
                                  index=range(self.df.shape[0]),
                                  columns=['beta', 'alpha'])
        return means

    def update_estim_beta(self, roll_scale_period):
        #df_train=df[:10]
        #x=df_train[tick2]
        #y=df_train[tick1]
        df = self.df.copy()
        states_param, kf, st_m, st_c = GetSpread.kalman_filter(self)
        states_param = GetSpread.udpate_kalman_filter(self, kf, st_m, st_c)
        df['beta'] = states_param['beta'].to_list()
        df['alpha'] = states_param['alpha'].to_list()
        df['misp_ptf'] = [df.reset_index().loc[i, self.tick1] - df.reset_index().loc[i, 'beta'] * df.reset_index().loc[i, self.tick2] - df.reset_index().loc[i, 'alpha'] for i in range(df.shape[0])]
        df['misp_ptf'] = [(df.reset_index().loc[i, 'misp_ptf'] - df.reset_index()['misp_ptf'].rolling(roll_scale_period).mean()[i])/df.reset_index()['misp_ptf'].rolling(roll_scale_period).std()[i] for i in range(df.shape[0])]
        return df