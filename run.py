import pandas as pd
import numpy as np
import json
from utils import utils
from metrics.metrics import *
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import sys

try:
    target = sys.argv[1]
    print(target)
    if target == 'test':
        dataset_dir = './testDatasets/'
    else:
        dataset_dir = './cleanedDatasets/'
except:
    dataset_dir = './cleanedDatasets/'

# define lists of models, datasets, and metrics
datasets = os.listdir(dataset_dir)
print(datasets)
model_names = ['logistic_regression', 'naive_bayes', 'nearest_neighbors', 'decision_tree',
               'random_forest', 'multilayer_perceptron', 'svm_model']
metrics = ['precision_range', 'max_parity_ratio', 'equalized_odds_diff',
           'overall_accuracy', 'FPR', 'F1', 'recall_range', 'brier_score_range']

# iterate through (model, dataset, metric) tuples
fairness_dict = defaultdict(dict)  # dict of resulting 3-D matrix
for dataset in datasets:
    # read in data (should already be cleaned and in standard format)
    dataset_path = os.path.join(dataset_dir, dataset)
    csv_path = list(
        filter(lambda f: f[-3:] == 'csv', os.listdir(dataset_path)))[0]
    json_path = list(
        filter(lambda f: f[-4:] == 'json', os.listdir(dataset_path)))[0]

    df = pd.read_csv(os.path.join(dataset_path, csv_path))
    with open(os.path.join(dataset_path, json_path), 'r') as f:
        config = json.load(f)

    dataset_name = config['dataset_name']
    print(dataset_name)

    # split X and y using columns from config
    # TODO: this try-catch should be removed, this is here to allow main to run even if some config files have errors
    try:
        X = df[config['X_cols']]
        y = df[config['y_col']]
        # TODO: handle multiple groups -- should be trivial, not worrying about it now
        groups = df[config['group_cols'][0]]
    except:
        print('\tError reading columns from ', dataset_name)
        continue

    # same as sklearn's train_test_split, but we include the column for the group
    X_train, X_test, y_train, y_test, group_train, group_test = utils.split_data(
        X, y, groups, train_pct=0.75)

    data_attributes = (X_train, X_test, y_train, y_test, group_train, group_test, config)

    for model_name in model_names:
        # apply model to dataset, yield result with predictions
        print('\t' + model_name)
        results, results_prob, mdl_obj = utils.run_models(
            model_name, *data_attributes)  # unpacking

        # apply fairness metric
        metric_dict = {}
        for metric in metrics:
            metric_dict[metric] = utils.apply_metric(
                metric, results, results_prob, mdl_obj, *data_attributes)
        fairness_dict[dataset_name][model_name] = metric_dict

output_df = pd.DataFrame.from_dict({(i, j): fairness_dict[i][j]
                                    for i in fairness_dict.keys()
                                    for j in fairness_dict[i].keys()},
                                   orient='index')
print(output_df)

if dataset_dir == './cleanedDatasets/':
    output_df.to_csv('fairness_df.csv')
