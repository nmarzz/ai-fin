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
# Dow 29 models
# model_paths = ['models/models/a2c_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/ddpg_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/ppo_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/sac_dow29_steps100000_start2000-01-01_end2018-01-01.model','models/models/td3_dow29_steps100000_start2000-01-01_end2018-01-01.model']
# NASDAQ paths
model_paths = ['models/models/a2c_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/ddpg_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/ppo_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/sac_nas29_steps1000000_start2005-01-01_end2018-11-28.model','models/models/td3_nas29_steps1000000_start2005-01-01_end2018-11-28.model']
start_date,split_date,data_type ,model = get_model_info_from_path(model_paths)

end_date = '2018-11-28' # Model is tested from split_date to end_date
split_date = '2016-10-01'
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

model_names = []
for path in model_paths:
    start_date,split_date,data_type ,model = get_model_info_from_path(path)
    model_names.append(model) # For plotting
    model_params = config.__dict__[f"{model.upper()}_PARAMS"]
    trained_model = agent.get_model(model,
                            model_kwargs = model_params,
                            verbose = 0).load(path)

    print('Testing...')
    df_account_value, df_actions = DRLAgent.average_predict(
        model=trained_model,
        environment = test_gym_env,n_evals = 10)

    df_account_value.to_csv('results/' + model +'account_values_10avg.csv')
    df_actions.to_csv('results/' + model +'actions_10avg.csv')
    dates_rl = matplotlib.dates.date2num(df_account_value['date'])

    plt.plot_date(dates_rl,df_account_value['account_value'],'-')


end_date = '2018-11-28' # Model is tested from split_date to end_date
split_date = '2016-10-01'
print('Comparing to Index')
dji = YahooDownloader(
            start_date=split_date, end_date=end_date, ticker_list=['^IXIC'] # Dow = '^DJI', NASDAQ = '^IXIC'
        ).fetch_data()
model_names.append('Nas.')
dates_rl = matplotlib.dates.date2num(df_account_value['date'])
dates_base = matplotlib.dates.date2num(dji['date'])


init_dji_shares = 1000000/dji['close'][0]

plt.plot_date(dates_base,dji['close'] * init_dji_shares,'-')
plt.legend(model_names)
plt.xticks(rotation=30)
plt.title(f'Models Trained from {start_date}-{split_date}')
plt.ylabel('Account Value')
plt.savefig(f'imgs/models_vs_NASDAQ_{split_date}_{end_date}.png')
