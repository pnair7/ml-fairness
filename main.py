import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from models.sklearn_models import logistic_regression, decision_tree
from utils import utils
import os

## define lists of models, datasets, and metrics
datasets = os.listdir('cleanedDatasets/')
models = [logistic_regression, decision_tree]
metrics = []

## iterate through (model, dataset, metric) tuples

for dataset in datasets:
    ## read in data (should already be cleaned and in standard format)
    dataset_path = os.path.join('cleanedDatasets', dataset)
    csv_path = list(filter(lambda f: f[-3:] == 'csv', os.listdir(dataset_path)))[0]
    json_path = list(filter(lambda f: f[-4:] == 'json', os.listdir(dataset_path)))[0]
    
    df = pd.read_csv(os.path.join(dataset_path, csv_path))
    with open(os.path.join(dataset_path, json_path), 'r') as f:
        config = json.load(f)

    print(config['dataset_name'])

    ## split X and y using columns from config
    ## TODO: this try-catch should be removed, this is here to allow main to run even if some config files have errors
    try:
        X = df[config['X_cols']]
        y = df[config['y_col']]
        groups = df[config['group_cols'][0]] # TODO: handle multiple groups -- should be trivial, not worrying about it now
    except:
        print('\tError reading columns from ', config['dataset_name'])
        continue

    # same as sklearn's train_test_split, but we include the column for the group
    X_train, X_test, y_train, y_test, group_train, group_test = utils.split_data(X, y, groups, train_pct=0.75)

    for model in models:
        ## apply model to dataset, yield result with predictions
        print(model(X_train, X_test, y_train, y_test, group_train, group_test, config))

        ## import functions for each fairness metric on standardized output format
        for metric in metrics:
            pass
            ## apply fairness metric to predictions
            
            ## output result (probably by appending to list that becomes result matrix)

# TODO: couple things to think about: multiple group columns? with/without using group in classification?