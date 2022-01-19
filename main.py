import pandas as pd
import numpy as np
import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os

## define lists of models, datasets, and metrics
datasets = os.listdir('cleanedDatasets/')
models = []
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

    ## split X and y using columns from config
    X = df[config['X_cols']]
    y = df[config['y_col']]

    # probably need to make a custom function to do this since we also need something for the sensitive groups
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

    for model in models:
        pass
    
    ## define model w/ parameters if needed

    ## apply model to dataset, yield result with predictions

    ## import functions for each fairness metric on standardized output format
    for metric in metrics:
        pass
        ## apply fairness metric to predictions
        
        ## output result (probably by appending to list that becomes result matrix

# TODO: couple things to think about: multiple group columns? with/without using group in classification?