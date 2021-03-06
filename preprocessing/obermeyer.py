import pandas as pd
from scipy.stats import percentileofscore
import json

data = pd.read_csv('rawDatasets/obermeyer_data.csv')

## HELPFUL DATA DICTIONARY: https://gitlab.com/labsysmed/dissecting-bias/-/blob/master/data/data_dictionary.md

# produce y column
data['refer'] = (data['risk_score_t'].apply(lambda x: percentileofscore(data['risk_score_t'], x))>= 97.0).astype(int)

# define list of predictor columns
predictors = list(filter(lambda x: x[-4:] == '_tm1', list(data.columns)))

# sample dataset json?
config = {
    'y_col' : 'refer', # which column are we predicting
    'X_cols' : predictors, # list of columns in X -- predictor variables
    'group_cols' : ['race', 'dem_female'], # what are the groups we're interested in being fair to?
    'prediction_type' : 'binary',
    'dataset_name' : 'Obermeyer Health Dataset',
    'data_path' : 'rawDatasets/obermeyer_data.csv',
    'data_script' : 'preprocessing/obermeyer.py'
}

# output to folder -- this structure could change
data.to_csv('cleanedDatasets/obermeyer/obermeyer_data.csv', index = False)
with open('cleanedDatasets/obermeyer/obermeyer_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
