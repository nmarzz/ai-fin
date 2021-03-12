import sys
sys.path.append("../FinRL-Library")

import pandas as pd
import warnings
import numpy as np
import os
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer


def get_train_dataset(datadir,data_type,start_date,end_date):

    if not data_type in ['dow30','crypto']:
        raise ValueError('Market type not supported')



    data_path = os.path.join(datadir,data_type + '.csv')    

    if not os.path.exists(data_path):
        if data_type == 'dow30':
            # If we don't have the data, we can download dow data from yahoo finance
            stock_tickers = config.DOW_30_TICKER
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
            df = fe.preprocess_data(df)
            df['log_volume'] = np.log(df.volume*df.close)
            df['change'] = (df.close-df.open)/df.close
            df['daily_variance'] = (df.high-df.low)/df.close
            df.to_csv(data_path,index = False)
        else:
            raise ValueError('Need to add crypto data to data directory')

    # Load and subset data
    full_df = pd.read_csv(data_path)
    max_date = max(full_df['date'])
    min_date = min(full_df['date'])


    if not (min_date == start_date):
        warnings.warn('Earliest possible start date is {}: You have chosen {}. The later date will be used'.format(min_date,start_date))
    if not (max_date == end_date):
        warnings.warn('Latest possible start date is {}: You have chosen {}. The earlier date will be used'.format(max_date,end_date))

    to_return = full_df[full_df['date'] >= start_date]
    to_return = to_return[to_return['date'] <= end_date]


    return to_return
