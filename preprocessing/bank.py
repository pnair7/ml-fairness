
import os 
import pandas as pd
import numpy as np
import json
from scipy.stats import percentileofscore
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('../rawDatasets/bank_data/bank-full.csv',  delimiter=';', quotechar='"')


print(data.columns)
# one-hot encode necessary columns
onehot_cols = [ 'job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'month', 'poutcome', ]
enc = OneHotEncoder(sparse=False, drop='if_binary')
enc.fit(data[onehot_cols])
one_hot_applied = pd.DataFrame(enc.transform(data[onehot_cols]), columns=enc.get_feature_names_out())
data = data.drop(columns=onehot_cols)
data = data.merge(one_hot_applied, left_index=True, right_index=True)


print(data.columns)
# produce y column
data.loc[(data['y'] =='no'), ['y']]= 0
data.loc[(data['y'] =='yes'), ['y']]= 1
data['y'] = (data['y'].astype(int))
print(data['y'].value_counts())

# get all columns except last one 
predictors = list(data.iloc[:,:-1])


# sample dataset json?
config = {
    'y_col' : 'y',
    'X_cols' : predictors,
    'group_cols' : [ 'marital_divorced', 'marital_married', 'marital_single'],
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
