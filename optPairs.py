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

class GetOptPair:
    def __init__(self, significance, df_prices):
        self.significance = significance
        self.df_prices = df_prices

    def clustering(self, X):
        list_k = range(2, 7)
        silhouettes = []
        for k in list_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='random')
            kmeans.fit(X)
            silhouettes.append(silhouette_score(X, kmeans.labels_))
        kl = KneeLocator(list_k, silhouettes, curve="convex", direction="decreasing")
        k_means = KMeans(n_clusters=kl.elbow)
        k_means.fit(X)
        clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
        # clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
        clustered_series = clustered_series[clustered_series != -1]
        return clustered_series

    def cointegration(self, cluster):
        pair_coin = []
        p_value = []
        n = cluster.shape[0]
        keys = cluster.keys()
        for i in range(n):
            for j in range(i + 1, n):
                asset_1 = self.df_prices[keys[i]]
                asset_2 = self.df_prices[keys[j]]
                results = sm.OLS(asset_1, asset_2).fit()
                # results = DynamicOLS(asset_1, asset_2).fit()
                # results = LinearRegression().fit(asset_2.to_numpy() , asset_1.to_numpy() )
                predict = results.predict(asset_2)
                error = asset_1 - predict
                ADFtest = ts.adfuller(error)
                if ADFtest[1] < self.significance:
                    pair_coin.append([keys[i], keys[j]])
                    p_value.append(ADFtest[1])
        return p_value, pair_coin

    def pair_selection(self, clustered_series, E_selection=False):
        Opt_pairs = []  # to get best pair in cluster i
        counts = clustered_series.value_counts()
        cluster_size_limit = 1000
        ticker_count_reduced = counts[(counts > 1) & (counts <= cluster_size_limit)]
        counts = clustered_series.value_counts()
        clusters_viz_list = list(counts[(counts < 500) & (counts > 1)].index)[::-1]
        if E_selection:  # select one pair from each cluster
            for i in clusters_viz_list:
                cluster = clustered_series[clustered_series == i]
                keys = cluster.keys()
                result = GetOptPair.cointegration(self, cluster)
                if len(result[0]) > 0:
                    if np.min(result[0]) < self.significance:
                        index = np.where(result[0] == np.min(result[0]))[0][0]
                        Opt_pairs.append([result[1][index][0], result[1][index][1]])
        else:
            p_value_contval = []
            pairs_contval = []
            for i in clusters_viz_list:
                cluster = clustered_series[clustered_series == i]
                result = GetOptPair.cointegration(self, cluster)
                if len(result[0]) > 0:
                    p_value_contval += result[0]
                    pairs_contval += result[1]

            Opt_pair_index = heapq.nsmallest(25, range(len(p_value_contval)), key=p_value_contval.__getitem__)
            Opt_pairs = operator.itemgetter(*Opt_pair_index)(pairs_contval)
        return Opt_pairs