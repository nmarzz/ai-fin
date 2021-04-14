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
# Dow
# model_paths = ['models/models/a2c_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/ddpg_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/ppo_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/sac_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/td3_dow29_steps100000_start2000-01-01_end2018-01-01.model']
# Dow without a2c
# model_paths = ['models/models/ddpg_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/ppo_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/sac_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/td3_dow29_steps100000_start2000-01-01_end2018-01-01.model']
# Nasdaq
model_paths = ['models/models/a2c_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/ddpg_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/ppo_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/sac_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/td3_nas29_steps1000000_start2005-01-01_end2018-11-28.model']
start_date,split_date,data_type ,model = get_model_info_from_path(model_paths)

end_date = '2020-12-31' # Model is tested from split_date to end_date

end_date = '2018-01-01' # Model is tested from split_date to end_date
split_date = '2015-01-01'
print(data_type)

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
df_account_value, df_actions = DRLAgent.average_predict(
    model=trained_model,
    environment = test_gym_env,n_evals = 15)



print('Comparing to DJI')
dji = YahooDownloader(
            start_date=split_date, end_date=end_date, ticker_list=['^DJI']
        ).fetch_data()
dates_rl = matplotlib.dates.date2num(df_account_value['date'])
dates_base = matplotlib.dates.date2num(dji['date'])


init_dji_shares = 1000000/dji['close'][0]

df_account_value.to_csv('results/binaverage_ensemble_accounts.csv')
df_actions.to_csv('results/binaverage_ensemble_actions.csv')

plt.plot_date(dates_rl,df_account_value['account_value'],'-')
plt.ylabel('Account Value')



# model_paths = 'models/models/ddpg_dow29_steps100000_start2000-01-01_end2018-01-01.model'
# model = 'ddpg'


if model == 'ensemble':
    trained_model = EnsembleModel(test_gym_env,model_paths,'average')
else:
    model_params = config.__dict__[f"{model.upper()}_PARAMS"]
    trained_model = agent.get_model(model,
                            model_kwargs = model_params,
                            verbose = 0).load(model_paths)



print('Testing...')
df_account_value, df_actions = DRLAgent.average_predict(
    model=trained_model,
    environment = test_gym_env,n_evals = 15)


df_account_value.to_csv('results/average_ensemble_accounts.csv')
df_actions.to_csv('results/average_ensemble_actions.csv')

print('Comparing to DJI')
dji = YahooDownloader(
            start_date=split_date, end_date=end_date, ticker_list=['^IXIC'] # Dow = '^DJI', NASDAQ = '^IXIC'
        ).fetch_data()
dates_rl = matplotlib.dates.date2num(df_account_value['date'])
dates_base = matplotlib.dates.date2num(dji['date'])


init_dji_shares = 1000000/dji['close'][0]

start_date = '2005-01-01'
split_date = '2018-11-28' 
plt.plot_date(dates_rl,df_account_value['account_value'],'-')
plt.plot_date(dates_base,dji['close'] * init_dji_shares,'-')
plt.legend(['bin-avg','avg','nasdaq'])
plt.title(f'Ensemble trained from {start_date}-{split_date}')
plt.ylabel('Account Value')
plt.xticks(rotation=30)
plt.savefig(f'imgs/{model}_vs_NASDAQ_{split_date}_{end_date}.png')
