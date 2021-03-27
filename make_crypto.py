import sys

import pandas as pd
import numpy as np
import os
import itertools
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer


indicators = config.TECHNICAL_INDICATORS_LIST

df = pd.read_csv('crypto-markets_top30.csv')

fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list = indicators,
                use_turbulence=True,
                user_defined_feature = False)




print('Adding Indicators')
processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)


processed_full.to_csv('data/crypto.csv',index = False)

print(processed_full.head(20))
