
import os 
import pandas as pd
import numpy as np
import json
from scipy.stats import percentileofscore

data = pd.read_csv('../rawDatasets/law_data.csv' )
# print(data.shape)
# print(data.head())

# Drop first column of dataframe
data = data.iloc[: , 1:]

print(data.shape)
print(data.head())



# produce y column
data['pass_bar'] = (data['pass_bar'].astype(int))
print(data['pass_bar'].value_counts())

# # get all columns except last one 
predictors = list(data.iloc[:,:-1])


# sample dataset json?
config = {
    'y_col' : 'default payment next month',
    'X_cols' : predictors,
    'group_cols' : ['y'],
    'prediction_type' : 'binary',
    'dataset_name' : 'race Dataset',
    'data_path' : 'rawDatasets/bank_data/bank-full.csv',
    'data_script' : 'preprocessing/bank.py'
}

# # output to folder  
# path = '../cleanedDatasets/bank'
# try: 
#     os.mkdir(path) 
#     print("New Folder has been created")
# except OSError as error: 
#     print(error)  

# data.to_csv('../cleanedDatasets/bank/bank_data.csv', index = False)
# with open('../cleanedDatasets/bank/bank_config.json', 'w', encoding='utf-8') as f:
#     json.dump(config, f, ensure_ascii=False, indent=4)
