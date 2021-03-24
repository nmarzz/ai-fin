# TODO: Automate model selection better
# Set up pipeline into ipynb notebook for backtesting with pyfolio

import argparse
import sys
import os
sys.path.append("../FinRL-Library")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

import sys
sys.path.append("../FinRL-Library")

import itertools
import pyfolio



parser = argparse.ArgumentParser(description='Test a RL Stock Trader')
parser.add_argument('--train-steps',type = int,default = 5000, metavar = 'TS')
parser.add_argument('--initial_investment',type = int,default = 1e6, metavar = 'INV')
parser.add_argument('--start-date',type = str,default = '2009-01-01', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--split-date',type = str,default = '2019-01-01', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--end-date',type = str,default = '2021-01-01', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--modeldir',type = str,default = 'models', metavar = 'STR')
parser.add_argument('--modelname',type = str, metavar = 'STR')
parser.add_argument('--datadir',type = str,default = 'data', metavar = 'STR')

args = parser.parse_args()



print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')


startdate = args.start_date
splitdate = args.split_date
enddate = args.end_date
train_steps = args.train_steps
df_name = os.path.join(args.datadir,'dow30_start{}_end{}.csv'.format(startdate,enddate))

stock_tickers = config.DOW_30_TICKER
indicators = config.TECHNICAL_INDICATORS_LIST

# Get data
df_train = get_dataset(args.datadir,'dow30',args.start_date,args.split_date)
df_test = get_dataset(args.datadir,'dow30',args.split_date,args.end_date)

stock_dimension = len(df_train.tic.unique())
state_space = 1 + 2*stock_dimension + len(indicators)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4

}

e_train_gym = StockTradingEnv(df = df_train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()


agent = DRLAgent(env = env_train)

args.model = 'ppo'
model_params = config.__dict__[f"{args.model.upper()}_PARAMS"]

model = agent.get_model(args.model,
                        model_kwargs = model_params,
                        verbose = 1)

print('Testing model')

args.modelname = 'models/models/ppo_dow30_steps15000_start2009-01-01_end2019-01-01.model'
trained_model = model.load(args.modelname)


e_trade_gym = StockTradingEnv(df = df_test,turbulence_threshold = 329, **env_kwargs)
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_model,
    environment = e_trade_gym)


print(df_account_value.head())

import matplotlib.pyplot as plt
import matplotlib

dji = YahooDownloader(
            start_date=args.split_date, end_date=args.end_date, ticker_list=['^DJI']
        ).fetch_data()


dates_rl = matplotlib.dates.date2num(df_account_value['date'])
dates_base = matplotlib.dates.date2num(dji['date'])


print(df_actions.head(10))


print('DJI at 0')
print(dji['close'][0])


plt.plot_date(dates_rl,df_account_value['account_value'],'-')
plt.plot_date(dates_base,dji['close'] * 42.8334494,'-')
plt.legend(['RL','DJI'])
plt.title('PPO model trained from 2010-2019')
plt.ylabel('Account Value')
plt.savefig('AFIG.png')
print(df_actions.head(20))
