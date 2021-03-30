from copy import deepcopy
import pandas as pd
import re
from finrl.config import config

def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)



def get_model_info_from_path(model_path):
    '''
    Utility function to return some aspects of the model for testing from how the model was saved

    '''
    error_msg = 'Could not parse model data from model name. Please manually enter the values'

    data_type = None
    model = None
    for type in config.SUPPORTED_DATA:
        if re.search(type,model_path):
            data_type = type

    for m in config.AVAILABLE_MODELS:
        if re.search(m,model_path):
            model = m

    dates = re.findall(r'\d{4}-\d{2}-\d{2}', model_path)

    if len(dates) < 2: # <2 allows to save model with day trained included (if it is last)
        raise ValueError(error_msg)
    elif data_type is None:
        raise ValueError(error_msg)
    elif model is None:
        raise ValueError(error_msg)

    return dates[0],dates[1],data_type,model
