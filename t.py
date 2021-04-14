import pandas as pd

data_path = 'data/dow290.csv'
full_df = pd.read_csv(data_path)


print(full_df.columns)


# full_df.to_csv('data/dow290.csv',index = False)
