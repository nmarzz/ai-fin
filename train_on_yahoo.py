# TODO:
# Make data folder if not exist

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



parser = argparse.ArgumentParser(description='Training RL Stock Traders')
parser.add_argument('--model', type=str, metavar='MOD',
                    help='Options [ppo,ddpg,a2c,td3,sac]')
parser.add_argument('--train-steps',type = int,default = 5000, metavar = 'TS')
parser.add_argument('--initial_investment',type = int,default = 1e6, metavar = 'INV')
parser.add_argument('--start-date',type = str,default = '2009-01-01', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--split-date',type = str,default = '2019-01-01', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--end-date',type = str,default = '2021-01-01', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--modeldir',type = str,default = 'models', metavar = 'STR')
parser.add_argument('--datadir',type = str,default = 'data', metavar = 'STR')

args = parser.parse_args()


args.model = 'ppo'
if not args.model in ['ppo','ddpg','a2c','td3','sac']:
    raise ValueError('Invalid model choice: must be one of [\'ppo\',\'ddpg\',\'a2c\',\'td3\',\'sac\']')



print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')


startdate = args.start_date
splitdate = args.split_date
enddate = args.end_date
train_steps = args.train_steps
modelName = '{}_dow30_steps{}_start{}_end{}.model'.format(args.model,train_steps,startdate,splitdate)
df_name = os.path.join(args.datadir,'dow30_start{}_end{}.csv'.format(startdate,enddate))

stock_tickers = config.DOW_30_TICKER
indicators = config.TECHNICAL_INDICATORS_LIST

# Get data
df_train = get_dataset(args.datadir,'dow30',args.start_date,args.split_date)


df_test = get_dataset(args.datadir,'dow30',args.split_date,args.end_date)

print(df_train.head())

d = df_train.loc[0,:]

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
model_params = config.__dict__[f"{args.model.upper()}_PARAMS"]

model = agent.get_model(args.model,
                        model_kwargs = model_params,
                        verbose = 1)

print('Training model')
model.learn(total_timesteps = train_steps,
            log_interval = 1,
            tb_log_name = '{}_{}'.format(modelName,datetime.datetime.now()))

model.save(os.path.join(args.modeldir,modelName))
