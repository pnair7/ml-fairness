import pandas as pd
import numpy as np
import json

data = pd.read_csv('../rawDatasets/crimedata.csv')
data = data.replace('?', np.nan).dropna(axis=0, subset=['ViolentCrimesPerPop']).dropna(axis=1)

# drop columns that are neither predictors nor labels
data = data.drop(columns=['state', 'fold', 'murders', 'murdPerPop', 'rapes',
                 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop'])

# as in https://arxiv.org/abs/1711.05144,use highest 30% of cities in violent crime per capita to predict "high crime" label
# also as suggested in https://arxiv.org/pdf/2110.00530.pdf, "divide the communities according to race
# by thresholding the attribute racepctblack(the percentage of the population that is African American) at 0.06"

data['ViolentCrimesPerPop'] = data['ViolentCrimesPerPop'].astype(float)
data['highcrime'] = (data['ViolentCrimesPerPop'].rank(pct=True) > 0.7).astype(int)

data['black'] = data['racepctblack'] > 6

# drop columns that are neither predictors nor labels
data = data.drop(columns=['ViolentCrimesPerPop', 'communityname', 'racepctblack', 'racePctWhite',
                 'racePctAsian', 'racePctHisp', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap', 'HispPerCap'])

predictors = list(data.columns[:-2])

config = {
    'y_col': 'highcrime',  # which column are we predicting
    'X_cols': predictors,  # list of columns in X -- predictor variables
    'group_cols': ['black'],  # what are the groups we're interested in being fair to?
    'prediction_type': 'binary',
    'dataset_name': 'Communities and Crime',
    'data_path': 'rawDatasets/crimedata.csv',
    'data_script': 'preprocessing/communities_and_crime.py'
}

data.to_csv('../cleanedDatasets/communities_and_crime/communities_and_crime.csv', index=False)
with open('../cleanedDatasets/communities_and_crime/crime_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
