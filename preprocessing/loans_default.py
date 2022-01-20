import pandas as pd
import numpy as np
import json
from scipy.stats import percentileofscore

data = pd.read_csv('../rawDatasets/loans_default_data.csv')

def clean_loans_default(df): 
  df = df.iloc[: , 1:] #removing ID column
  df = df.rename(columns=df.iloc[0]).drop(df.index[0]) #making first row the header names
  df = df.reset_index(drop=True)
  return df

df3 = clean_loans_default(data) 
data = df3 


# produce y column
data['default payment next month'] = (data['default payment next month'].astype(int))
print(data.head() )


# get all columns except last one 
predictors = list(data.iloc[:,:-1])


# sample dataset json?
config = {
    'y_col' : 'default payment next month',
    'X_cols' : predictors,
    'group_cols' : ['SEX'],
    'prediction_type' : 'binary',
    'dataset_name' : 'Loans Default Dataset',
    'data_path' : 'rawDatasets/loans_default_data.csv',
    'data_script' : 'preprocessing/loans_default.py'
}

# output to folder  
data.to_csv('../cleanedDatasets/loans_default/loans_default_data.csv', index = False)
with open('../cleanedDatasets/loans_default/loans_default_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
