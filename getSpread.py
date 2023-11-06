import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
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

        # Vérification de la taille des données
        if len(x) < 10 or len(y) < 10:
            raise ValueError("Insufficient data for tick1 and/or tick2. Please provide at least 10 observations.")

        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                        initial_state_mean=[0, 0],
                        initial_state_covariance=np.ones((2, 2)),
                        transition_matrices=np.eye(2),
                        observation_matrices=obs_mat,
                        observation_covariance=2,
                        transition_covariance=trans_cov)
        state_means, state_covs = kf.filter(y.values)
        res_param = pd.DataFrame(state_means,
                                index=range(len(x)),
                                columns=['beta', 'alpha'])
        return res_param, kf, state_means, state_covs


    def update_kalman_filter(self, kf, history_state_means, history_state_covs):
        observations = self.df[self.tick1].values
        new_state_means = np.zeros((self.df.shape[0], 2))
        new_state_covs = np.zeros((self.df.shape[0], 2, 2))
        
        for i in range(self.df.shape[0]):
            obs_mat = np.asarray([[self.df[self.tick2].iloc[i], 1]])
            new_state_means[i], new_state_covs[i] = kf.filter_update(history_state_means[-1],
                                                                    history_state_covs[-1],
                                                                    observation=observations[i],
                                                                    observation_matrix=obs_mat)
            history_state_means = np.concatenate([history_state_means, [new_state_means[i]]], axis=0)
            history_state_covs = np.concatenate([history_state_covs, [new_state_covs[i]]], axis=0)

        means = pd.DataFrame(new_state_means, index=range(self.df.shape[0]), columns=['beta', 'alpha'])
        return means


    def update_estim_beta(self, roll_scale_period):
        df = self.df.copy()
        states_param, kf, st_m, st_c = self.kalman_filter()
        states_param = self.update_kalman_filter(kf, st_m, st_c)
        df['beta'] = states_param['beta'].to_list()
        df['alpha'] = states_param['alpha'].to_list()
        df['misp_ptf'] = df[self.tick1] - df['beta'] * df[self.tick2] - df['alpha']
        misp_mean = df['misp_ptf'].rolling(roll_scale_period).mean()
        misp_std = df['misp_ptf'].rolling(roll_scale_period).std()
        df['misp_ptf'] = (df['misp_ptf'] - misp_mean) / misp_std
        return df
