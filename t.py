# TODO: Automate model selection better
# Set up pipeline into ipynb notebook for backtesting with pyfolio

import argparse
import sys
import os


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import pyfolio
import re
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent,DRLEnsembleAgent
from finrl.trade.backtest import backtest_stats, get_baseline, backtest_plot
from pprint import pprint

from utils.enviroments import StockTradingEnvV2
from utils.data_utils import get_dataset
from utils.preprocess import get_model_info_from_path
from utils.models import EnsembleModel



data_dir = 'data'
# model_paths = ['models/models/a2c_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/ddpg_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/ppo_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/sac_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/td3_dow29_steps100000_start2000-01-01_end2018-01-01.model']
model_paths = 'models/models/ppo_dow29_steps100000_start2000-01-01_end2018-01-01.model'
start_date,split_date,data_type ,model = get_model_info_from_path(model_paths)

end_date = '2020-12-31' # Model is tested from split_date to end_date


# Get data
df = get_dataset(data_dir,data_type,split_date,end_date)

print(f'Testing from {start_date} to {end_date}')

stock_dimension = len(df.tic.unique())
indicators = config.TECHNICAL_INDICATORS_LIST

state_space = 1 + 2*stock_dimension + len(indicators)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 500,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

test_gym_env = StockTradingEnv(df = df,turbulence_threshold = 329, **env_kwargs)
agent = DRLAgent(env = test_gym_env)


if model == 'ensemble':
    trained_model = EnsembleModel(test_gym_env,model_paths,'binaverage')
else:
    model_params = config.__dict__[f"{model.upper()}_PARAMS"]
    trained_model = agent.get_model(model,
                            model_kwargs = model_params,
                            verbose = 0).load(model_paths)



print('Testing...')
# df_account_value, df_actions = DRLAgent.DRL_prediction(
#     model=trained_model,
#     environment = test_gym_env)

#
# print(df_account_value.shape)
#
# print(df_account_value.head())
# df_account_value, df_actions = DRLAgent.DRL_prediction(
#     model=trained_model,
#     environment = test_gym_env)
#
# df_account_value['new'] = df_account_value['account_value'] + df_account_value['account_value']

def average_predict(model,environment,n_evals=2):
    values = []
    actions = []
    for i in range(n_evals):
        account_value, action_value = DRLAgent.DRL_prediction(model,environment)
        values.append(account_value)
        actions.append(action_value)

    value_sum = values[0]['account_value']
    action_sum = actions[0]
    for i in range(1,n_evals):
        value_sum += values[i]['account_value']
        action_sum += actions[i]

    values[0]['account_value'] = value_sum / n_evals
    return_values = values[0].set_index('date')

    action_sum /= n_evals

    return return_values , action_sum






values,actions = average_predict(trained_model,test_gym_env,n_evals = 1)
values.index = pd.to_datetime(values.index)


pyfolio.create_simple_tear_sheet(values['account_value'])
