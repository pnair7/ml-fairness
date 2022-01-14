import pandas as pd
from scipy.stats import percentileofscore

data = pd.read_csv('rawDatasets/obermeyer_data.csv')

## HELPFUL DATA DICTIONARY: https://gitlab.com/labsysmed/dissecting-bias/-/blob/master/data/data_dictionary.md

# produce y column
data['refer'] = (data['risk_score_t'].apply(lambda x: percentileofscore(data['risk_score_t'], x))>= 97.0).astype(int)

# drop current time columns
predictors = list(filter(lambda x: x[-4:] == '_tm1', list(data.columns)))

print(predictors)

# sample dataset json?
# {
#     'y_col' : 'refer_t',
#     'X_cols' : predictors,
#     'group_cols' : ['race', 'dem_female'],
#     'prediction_type' : '',
#     'dataset_name' : 'Obermeyer Health Dataset',
#     'data_path' : 'rawDatasets/obermeyer_data.csv',
#     'data_script' : 'preprocessing/obermeyer.py'
# }
