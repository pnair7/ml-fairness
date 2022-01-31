import pandas as pd
import numpy as np
import json

h = ['age', 'workclass', 'fnlwgt', 'education', 'years-of-education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'Salary']
data = pd.read_csv('.../rawDatasets/adult_census_data.txt', names=h)
data = pd.get_dummies(data, columns=['marital-status', 'workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
data['above_50k'] = data['Salary'].apply(lambda x: int(x == " >50K"))
data = data.drop(columns=['Salary'])

predictors = list(data.iloc[:,:-1])

config = {
    'y_col' : 'above_50k', # which column are we predicting
    'X_cols' : predictors, # list of columns in X -- predictor variables
    'group_cols' : ['race_ Black'], # what are the groups we're interested in being fair to?
    'prediction_type' : 'binary',
    'dataset_name' : 'Adult Data Set (Census)',
    'data_path' : 'rawDatasets/adult_ceusus_data.txt',
    'data_script' : 'preprocessing/adult_census.py'
}

data.to_csv('.../cleanedDatasets/adult_census/adult_census_data.csv', index = False)
with open('.../cleanedDatasets/adult_census/adult_census_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
