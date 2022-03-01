
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

# print(data.shape)
# print(data.head())



# produce y column
data['pass_bar'] = (data['pass_bar'].astype(int))
print(data['pass_bar'].value_counts())

# # get all columns except last one 
predictors = list(data.iloc[:,:-1])


# sample dataset json?
config = {
    'y_col' : 'pass_bar',
    'X_cols' : predictors,
    'group_cols' : ['male', 'racetxt'],
    'prediction_type' : 'binary',
    'dataset_name' : 'law Dataset',
    'data_path' : 'rawDatasets/rawDatasets/law_data.csv',
    'data_script' : 'preprocessing/law.py'
}

# output to folder  
path = '../cleanedDatasets/law'
try: 
    os.mkdir(path) 
    print("New Folder has been created under cleaned datasets for law data")
except OSError as error: 
    print(error)  

data.to_csv('../cleanedDatasets/law/law_data.csv', index = False)
with open('../cleanedDatasets/law/law_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
