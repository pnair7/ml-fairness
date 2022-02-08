
import os 
import pandas as pd
import numpy as np
import json
from scipy.stats import percentileofscore

data = pd.read_csv('../rawDatasets/bank_data/bank-full.csv',  delimiter=';', quotechar='"')


# produce y column
data.loc[(data['y'] =='no'), ['y']]= 0
data.loc[(data['y'] =='yes'), ['y']]= 1
data['y'] = (data['y'].astype(int))
print(data['y'].value_counts())

# get all columns except last one 
predictors = list(data.iloc[:,:-1])


# sample dataset json?
config = {
    'y_col' : 'default payment next month',
    'X_cols' : predictors,
    'group_cols' : ['y'],
    'prediction_type' : 'binary',
    'dataset_name' : 'Bank Dataset',
    'data_path' : 'rawDatasets/bank_data/bank-full.csv',
    'data_script' : 'preprocessing/bank.py'
}

# output to folder  
path = '../cleanedDatasets/bank'
try: 
    os.mkdir(path) 
    print("New Folder has been created")
except OSError as error: 
    print(error)  

data.to_csv('../cleanedDatasets/bank/bank_data.csv', index = False)
with open('../cleanedDatasets/bank/bank_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
