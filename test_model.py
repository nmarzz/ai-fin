# TODO:
# Make data folder if not exist
# Raise error or warning when data selection dates are not sharp

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
import utils.data_utils
import itertools
import pyfolio



parser = argparse.ArgumentParser(description='Testing RL Stock Traders')
parser.add_argument('--model', type=str, metavar='MOD',
                    help='Options [ppo,ddpg,a2c,td3,sac]')
parser.add_argument('--data-type',type = str,metavar = 'DTYPE',help = 'one of: dow30 or crypto')
parser.add_argument('--train-steps',type = int,default = 5000, metavar = 'TS')
parser.add_argument('--initial_investment',type = int,default = 1e6, metavar = 'INV')
parser.add_argument('--start-date',type = str,default = '2009-01-01', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--end-date',type = str,default = '2021-01-19', metavar = 'STR',help = 'expects format YYYY-MM-DD')
parser.add_argument('--modeldir',type = str,default = 'models', metavar = 'STR')
parser.add_argument('--datadir',type = str,default = 'data', metavar = 'STR')

args = parser.parse_args()


if not args.model in ['ppo','ddpg','a2c','td3','sac']:
    raise ValueError('Invalid model choice: must be one of [\'ppo\',\'ddpg\',\'a2c\',\'td3\',\'sac\']')
if not args.data_type in ['dow30','crypto']:
    raise ValueError('Invalid data choice: must be one of [\'dow30\',\'crypto\']')

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')


startdate = args.start_date
enddate = args.end_date
train_steps = args.train_steps
modelName = '{}_{}_steps{}_start{}_end{}.model'.format(args.model,args.data_type,train_steps,startdate,enddate)

# get testing data
data = utils.data_utils.get_dataset(args.datadir,args.data_type,args.start_date,args.end_date)

# get model

# make testing enviroment

# test model

# save testing results




# # Get data
# if os.path.exists(df_name):
#     df = pd.read_csv(df_name)
# else:
#     print('Getting Data: ')
#     df = YahooDownloader(start_date = startdate,
#                          end_date = enddate,
#                          ticker_list = stock_tickers).fetch_data()
#
#     fe = FeatureEngineer(
#                     use_technical_indicator=True,
#                     tech_indicator_list = indicators,
#                     use_turbulence=True,
#                     user_defined_feature = False)
#
#     print('Adding Indicators')
#     df = fe.preprocess_data(df)
#     df['log_volume'] = np.log(df.volume*df.close)
#     df['change'] = (df.close-df.open)/df.close
#     df['daily_variance'] = (df.high-df.low)/df.close
#     df.to_csv(df_name,index = False)
#
#
# # Now define the training enviroment
# df_train = data_split(df, startdate,splitdate)
# df_test = data_split(df, splitdate,enddate)
#
# information_cols = ['daily_variance', 'change', 'log_volume', 'close','day',
#                     'macd', 'rsi_30', 'cci_30', 'dx_30', 'turbulence']
#
# initial_investment = 1e6
# train_gym = StockTradingEnvV2(df = df_train,initial_amount = initial_investment,hmax = 5000,
#                                 out_of_cash_penalty = 0,
#                                 cache_indicator_data=False,
#                                 cash_penalty_proportion=0.2,
#                                 reward_scaling=1,
#                                 daily_information_cols = information_cols,
#                                 print_verbosity = 500, random_start = True)
#
# test_gym = StockTradingEnvV2(df = df_test,initial_amount = initial_investment,hmax = 5000,
#                                 out_of_cash_penalty = 0,
#                                 cash_penalty_proportion=0.2,
#                                 reward_scaling = 1,
#                                 cache_indicator_data=False,
#                                 daily_information_cols = information_cols,
#                                 print_verbosity = 500, random_start = False)
#
# test_gym.seed(1331)
#
#
#
# # this is our training env. It allows multiprocessing
# env_train, _ = train_gym.get_multiproc_env(n = n_cores)
# env_trade, _ = test_gym.get_sb_env()
#
# agent = DRLAgent(env = env_train)
#
# ppo_params ={'n_steps': 256,
#              'ent_coef': 0.01,
#              'learning_rate': 0.00009,
#              'batch_size': 512,
#             'gamma': 0.99}
# policy_kwargs = {
#     "net_arch": [1024, 1024,1024, 1024,  1024],
# }
# model = agent.get_model("ppo",
#                         model_kwargs = ppo_params,
#                         policy_kwargs = policy_kwargs, verbose = 1)
# model = model.load(modelName)
#
# ## Now we begin testing the model
# df_account_value, df_actions = DRLAgent.DRL_prediction(model=model,environment = test_gym)
# backtest_results = backtest_stats(account_value=df_account_value, value_col_name = 'total_assets')
#
# def backtest(account_values,baseline):
#     pass
#
# baseline_ticker = "^DJI"
# baseline_dji = YahooDownloader(
#             start_date= startdate, end_date=enddate, ticker_list=[baseline_ticker]
#         ).fetch_data()
#
# test_returns = get_daily_return(df_account_value, value_col_name='total_assets')
# baseline_returns = get_daily_return(baseline_dji, value_col_name='close')
#
#
#
# # plt.plot(test_returns)
# # plt.plot(baseline_returns)
# # plt.legend(['us','dji'])
# # plt.savefig("mygraph.png")
#
# test_returns.to_csv('f.csv')
# baseline_returns.to_csv('me.csv')
# #
# # with pyfolio.plotting.plotting_context(context = "paper",font_scale=1.1):
# #         pyfolio.create_full_tear_sheet(
# #             returns=test_returns, benchmark_rets=baseline_returns, set_context=False
# #         )
#
#
# # backtest_plot(df_account_value,
# #              baseline_ticker = '^DJI',
# #              baseline_start = '2019-01-01',
# #              baseline_end = '2021-01-01', value_col_name = 'total_assets')
