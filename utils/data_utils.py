import sys
sys.path.append("../FinRL-Library")

import pandas as pd
import warnings
import numpy as np
import os
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
import itertools

def get_dataset(datadir,data_type,start_date,end_date):

    if not data_type in ['dow29','crypto']:
        raise ValueError('Market type not supported')


    data_path = os.path.join(datadir,data_type + '.csv')

    if not os.path.exists(data_path):
        if data_type == 'dow29':
            # If we don't have the data, we can download dow data from yahoo finance
            stock_tickers = config.DOW_30_TICKER_MINUS_VISA
            indicators = config.TECHNICAL_INDICATORS_LIST
            print('Getting Data: ')
            df = YahooDownloader(start_date = '2000-01-01',
                                 end_date = '2021-01-01',
                                 ticker_list = stock_tickers).fetch_data()

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
            processed.to_csv(data_path,index = False)
        else:
            raise ValueError('Need to add crypto data to data directory')

    # Load and subset data
    full_df = pd.read_csv(data_path)
    max_date = max(full_df['date'])
    min_date = min(full_df['date'])


    if not (min_date <= start_date):
        warnings.warn('Earliest possible start date is {}: You have chosen {}. The later date will be used'.format(min_date,start_date))
    if not (max_date >= end_date):
        warnings.warn('Latest possible end date is {}: You have chosen {}. The earlier date will be used'.format(max_date,end_date))

    to_return = data_split(full_df,start_date,end_date)


    return to_return



def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data
