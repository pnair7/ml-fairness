import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json

# import data
data = pd.read_csv('../rawDatasets/diabetic_data.csv')

# reduce size
data = data.sample(n=20000).reset_index(drop=True)

# create label
data['readmitted'] = (data['readmitted'] == '<30').astype(int)
data = data.rename(columns={'readmitted': 'readmitted_30'})

# remove unnecessary/missing data
data = data.drop(columns=['diag_1', 'diag_2', 'diag_3'])
data = data[data['race'] != '?']

# one-hot encode necessary columns
onehot_cols = ['age', 'race', 'max_glu_serum', 'A1Cresult',
               'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
               'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
               'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
               'tolazamide', 'examide', 'citoglipton', 'insulin',
               'glyburide-metformin', 'glipizide-metformin',
               'glimepiride-pioglitazone', 'metformin-rosiglitazone',
               'metformin-pioglitazone', 'change', 'diabetesMed']
enc = OneHotEncoder(sparse=False)
enc.fit(data[onehot_cols])
one_hot_applied = pd.DataFrame(enc.transform(data[onehot_cols]), columns=enc.get_feature_names_out())
data = data.drop(columns=onehot_cols)
data = data.merge(one_hot_applied, left_index=True, right_index=True)

predictors = list(data.columns)
predictors.remove('gender')

config = {
    'y_col': 'readmitted_30',  # which column are we predicting
    'X_cols': predictors,  # list of columns in X -- predictor variables
    'group_cols': ['gender'],  # what are the groups we're interested in being fair to?
    'prediction_type': 'binary',
    'dataset_name': 'Diabetes Dataset 1999-2008',
    'data_path': 'rawDatasets/diabetic_data.csv',
    'data_script': 'preprocessing/diabetic.py'
}

# output to folder -- this structure could change
data.to_csv('../cleanedDatasets/diabetic/diabetic_data.csv', index=False)
with open('../cleanedDatasets/diabetic/diabetic_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
