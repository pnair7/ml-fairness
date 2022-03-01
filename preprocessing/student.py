import pandas as pd
import json
import os

data = pd.read_csv('rawDatasets/student-mat.csv', sep=';')
data.drop(columns=['famsize', 'reason', 'guardian', 'nursery', 'romantic', 'G2', 'G3'], inplace=True)
data = pd.get_dummies(data, columns=['school', 'sex', 'address', 'address', 'Pstatus',
                      'Mjob', 'Fjob', 'schoolsup', 'famsup', 'paid', 'activities', 'higher', 'internet'])
data['high_grade'] = data['G1'].apply(lambda x: int(x >= 12))
data.drop(columns=['G1'], inplace=True)

predictors = list(data.iloc[:,:-1])

config = {
    'y_col': 'high_grade',  # which column are we predicting
    'X_cols': predictors,  # list of columns in X -- predictor variables
    'group_cols': ['sex_F'],  # what are the groups we're interested in being fair to?
    'prediction_type': 'binary',
    'dataset_name': 'Student Performance Datset',
    'data_path': 'rawDatasets/student_mat.csv',
    'data_script': 'preprocessing/student.py'
}

path = 'cleanedDatasets/student'
try:
    os.mkdir(path)
    print("New Folder has been created")
except OSError as error:
    print(error)


data.to_csv('cleanedDatasets/student/student_data.csv', index=False)
with open('cleanedDatasets/student/student_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
