import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from sklearn.preprocessing import StandardScaler
from pandas_datareader import data as web
from backtest import Backtest
from optPairs import GetOptPair
from getSpread import GetSpread

import warnings

from data import GetData
warnings.simplefilter('ignore')

class OptimizeSignals:
    @staticmethod
    def objective(params):
        _, final_pnl, _ = Backtest.simple_backtest(params['data'], [params['const_coeff'], params['const_std'], int(params['roll_vol'])])
        return -1.0 * final_pnl

    @staticmethod
    def bayesian_opti(objective, space_param, max_evals):
        # Algorithm
        tpe_algorithm = tpe.suggest

        # Trials object to track progress
        bayes_trials = Trials()

        # Optimize
        best = fmin(fn=objective, space=space_param, algo=tpe_algorithm, max_evals=max_evals, trials=bayes_trials,
                    return_argmin=False)
        df_best = pd.DataFrame.from_dict(best, orient='index')
        print(bayes_trials.best_trial['result']['loss'])
        return df_best, bayes_trials.best_trial['result']['loss']

    @staticmethod
    def multi_bay_opti(objective, space_param, nb=1):
        final = pd.DataFrame(
            columns=['const_coeff', 'const_std', 'roll_vol', 'SCORE'], index=range(nb))
        for i in range(nb):
            optim = OptimizeSignals.bayesian_opti(objective, space_param, 50)
            final['const_std'].iloc[i] = optim[0].loc['const_std'][0]
            final['roll_vol'].iloc[i] = optim[0].loc['roll_vol'][0]
            final['const_coeff'].iloc[i] = optim[0].loc['const_coeff'][0]
            final['SCORE'].iloc[i] = optim[1]
        return final

    @staticmethod
    def grid_search(df):
        #570 iterations
        space_param0 = np.arange(0.5, 2.5, 0.2)
        space_param1 = np.arange(0.1, 1, 0.2)
        space_param2 = np.arange(15, 55, 10)
        pnl_res = []
        val_param0 = []
        val_param1 = []
        val_param2 = []
        print(f'nb iterations : {len(space_param0)*len(space_param1)*len(space_param2)}')
        for i in space_param0:
            for j in space_param1:
                for h in space_param2:
                    pnl_res.append(Backtest.simple_backtest(df, [i, j, h])[1])
                    val_param0.append(i)
                    val_param1.append(j)
                    val_param2.append(h)

        idx = pnl_res.index(max(pnl_res))
        return pnl_res[idx], val_param0[idx], val_param1[idx], val_param2[idx]


if __name__ == '__main__':
    # PARTIES SP500
    indices = ['^GSPC', '^DJI', '^RUT', '^VIX', '^FCHI', '^FTSE', '^BUK100P', '^N225']
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_list = np.array(sp500[0]['Symbol'])
    cac40 = pd.read_html('https://en.wikipedia.org/wiki/CAC_40#Composition')
    cac40_list = np.array(cac40[4]['Ticker'])
    full_tickers = list(sp500)
    start = dt.datetime(2017, 11, 9)
    end = dt.datetime(2023, 5, 22)
    get_data = GetData(start, end)
    data = get_data.dl_close_data(full_tickers, full_tickers)

    df_nan = pd.DataFrame(data.isnull().sum(), columns=['nb_nan'])
    l_tick_to_del = df_nan.sort_values('nb_nan')[-18:].reset_index()['ticker'].to_list()
    data = data.drop(columns=l_tick_to_del)
    df_prices = data.fillna(method='ffill')
    returns = data.pct_change().mean()*252
    returns = pd.DataFrame(returns)
    returns.columns = ['returns']
    returns['volatility'] = data.pct_change().std()*np.sqrt(252)
    df_returns = returns.copy()
    scale = StandardScaler().fit(df_returns)
    scaled_data = pd.DataFrame(scale.fit_transform(df_returns),columns=df_returns.columns, index=df_returns.index)

    significance = 0.05
    find_opt_pairs = GetOptPair(significance, df_prices)
    clustered_series = find_opt_pairs.clustering(scaled_data)
    opt_pairs = find_opt_pairs.pair_selection(clustered_series)
    print("Number of cointegrated pairs: ", len(opt_pairs))
    print("Pairs with lowest p-value among all the clusters:", opt_pairs)
    all_pairs = []
    for i in range(len(opt_pairs)):
        all_pairs.append(opt_pairs[i][0])
        all_pairs.append(opt_pairs[i][1])
    all_pairs = np.unique(all_pairs)
    final_data = data[all_pairs]

    full_df = final_data.dropna().copy()
    dict_df_res_bay = {}
    dict_mx_bnf_bay = {}
    dict_bnf_bay = {}
    dict_full_bnf_bay = {}
    for elem in opt_pairs:
        this_df = full_df[elem]
        name = elem[0] + '_' + elem[1]
        this_df.columns = ['btc_close', 'eth_close']
        df_train = this_df[700:1600]
        df_test = this_df[1600:]
        get_spread_train = GetSpread(df_train, 'btc_close', 'eth_close')
        df_train = get_spread_train.update_estim_beta(50)
        space_param = {
            "const_coeff": hp.choice("const_coeff", np.arange(0.5, 2, 0.01)),
            "const_std": hp.choice("const_std", np.arange(0, 2.5, 0.01)),
            "roll_vol": hp.choice("roll_vol", np.arange(5, 150, 5, dtype=int)),
            "data": df_train.reset_index()
        }
        opti = OptimizeSignals.multi_bay_opti(OptimizeSignals.objective, space_param, 2)
        get_spread_test = GetSpread(df_test, 'btc_close', 'eth_close')
        df_test = get_spread_test.update_estim_beta(50)
        df_res, mx_bnf, bnf = Backtest.simple_backtest(df_test.reset_index(), params=[opti.sort_values('SCORE').reset_index()['const_coeff'][0],
                                               opti.sort_values('SCORE').reset_index()['const_std'][0],
                                               int(opti.sort_values('SCORE').reset_index()['roll_vol'][0])])
        print(f'Pnl on test data : {round(mx_bnf, 2)} for strat {name}')
        dict_df_res_bay[name] = df_res
        dict_mx_bnf_bay[name] = mx_bnf
        dict_bnf_bay[name] = bnf
        dict_full_bnf_bay[name] = df_res['pnl_strat'].values

    full_return_ptf_bay = pd.DataFrame.from_dict(dict_full_bnf_bay)
    results_df_bay = full_return_ptf_bay / len(full_return_ptf_bay.columns)
    final_res_bay = results_df_bay.sum(axis=1)
    plt.figure(figsize=(15, 10))
    plt.title('Comparison of the cumulative returns of the different strategies')
    plt.plot(final_res_bay[50:].to_list(), label='Bayesian Search')
    plt.show()

    # PARTIES CRYPTO
    crypto = ['BTC/USD', 'ETH/USD', 'BNB/USD', 'XRP/USD', 'ADA/USD', 'TRX/USD', 'LTC/USD', 'LINK/USD']
    data = GetData.twelve_data(crypto, '1h')
    returns = data.pct_change().mean()*252
    returns = pd.DataFrame(returns)
    returns.columns = ['returns']
    returns['volatility'] = data.pct_change().std()*np.sqrt(252)
    df_returns = returns.copy()
    scale = StandardScaler().fit(df_returns)
    scaled_data = pd.DataFrame(scale.fit_transform(df_returns),columns=df_returns.columns, index=df_returns.index)

    significance = 0.05
    find_opt_pairs = GetOptPair(significance, data)
    clustered_series = find_opt_pairs.clustering(scaled_data)
    opt_pairs = find_opt_pairs.pair_selection(clustered_series)
    print("Number of cointegrated pairs: ", len(opt_pairs))
    print("Pairs with lowest p-value among all the clusters:", opt_pairs)
    all_pairs = []
    for i in range(len(opt_pairs)):
        all_pairs.append(opt_pairs[i][0])
        all_pairs.append(opt_pairs[i][1])
    all_pairs = np.unique(all_pairs)
    final_data = data[all_pairs]

    full_df = final_data.dropna().copy()
    dict_df_res_bay = {}
    dict_mx_bnf_bay = {}
    dict_bnf_bay = {}
    dict_full_bnf_bay = {}
    for elem in opt_pairs:
        this_df = full_df[elem]
        name = elem[0] + '_' + elem[1]
        this_df.columns = ['btc_close', 'eth_close']
        df_train = this_df[:3000]
        df_test = this_df[3000:].reset_index(drop=True)
        get_spread_train = GetSpread(df_train, 'btc_close', 'eth_close')
        df_train = get_spread_train.update_estim_beta(50)
        space_param = {
            "const_coeff": hp.choice("const_coeff", np.arange(0.5, 2, 0.01)),
            "const_std": hp.choice("const_std", np.arange(0, 2.5, 0.01)),
            "roll_vol": hp.choice("roll_vol", np.arange(5, 150, 5, dtype=int)),
            "data": df_train.reset_index()
        }
        opti = OptimizeSignals.multi_bay_opti(OptimizeSignals.objective, space_param, 2)
        get_spread_test = GetSpread(df_test, 'btc_close', 'eth_close')
        df_test = get_spread_test.update_estim_beta(50)
        df_res, mx_bnf, bnf = Backtest.simple_backtest(df_test.reset_index(), params=[
            opti.sort_values('SCORE').reset_index()['const_coeff'][0],
            opti.sort_values('SCORE').reset_index()['const_std'][0],
            int(opti.sort_values('SCORE').reset_index()['roll_vol'][0])])
        print(f'Pnl on test data : {round(mx_bnf, 2)} for strat {name}')
        dict_df_res_bay[name] = df_res
        dict_mx_bnf_bay[name] = mx_bnf
        dict_bnf_bay[name] = bnf
        dict_full_bnf_bay[name] = df_res['pnl_strat'].values

    full_return_ptf_bay = pd.DataFrame.from_dict(dict_full_bnf_bay)
    results_df_bay = full_return_ptf_bay / len(full_return_ptf_bay.columns)
    final_res_bay = results_df_bay.sum(axis=1)
    plt.figure(figsize=(15, 10))
    plt.title('Comparison of the cumulative returns of the different strategies')
    plt.plot(final_res_bay[50:].to_list(), label='Bayesian Search')
    plt.show()

    return_final_ptf = np.log(final_res_bay/final_res_bay.shift(1))