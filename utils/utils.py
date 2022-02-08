import pandas as pd
import numpy as np
import random
from models.sklearn_models import *
from metrics.metrics import *


def split_data(X, y, groups, train_pct=0.75):
    '''
    X and y are Pandas DataFrames of features and labels, respectively. (See ~/main.py).
    groups is a Series of the group column.
    Requires all three to share a common index -- which they should, since they're generated from
    the same source DataFrame. But if this breaks in the future, maybe that's why.
    '''
    assert len(X) == len(y) == len(groups)

    random.seed(42)
    X_shuffled = X.sample(frac=1)
    y_shuffled = y.loc[list(X_shuffled.index)]
    groups = y.loc[list(X_shuffled.index)]

    # generate train, test, groups split by train_size
    n_train = int(train_pct * len(X))
    X_train = X_shuffled.iloc[:n_train]
    X_test = X_shuffled.iloc[n_train:]
    y_train = y_shuffled.iloc[:n_train]
    y_test = y_shuffled.iloc[n_train:]
    group_train = groups.iloc[:n_train]
    group_test = groups.iloc[n_train:]

    return X_train, X_test, y_train, y_test, group_train, group_test


def run_models(model_name, X_train, X_test, y_train, y_test, group_train, group_test, config):
    model_params = (X_train, X_test, y_train, y_test, group_train, group_test, config)
    if model_name == 'logistic_regression':
        return logistic_regression(*model_params)
    elif model_name == 'decision_tree':
        return decision_tree(*model_params)
    elif model_name == 'random_forest':
        return random_forest(*model_params)
    elif model_name == 'multilayer_perceptron':
        return multilayer_perceptron(*model_params)
    elif model_name == 'svm_model':
        return svm_model(*model_params)


def apply_metric(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    metric_params = (metric_name, results, results_prob, mdl_obj, X_train,
                     X_test, y_train, y_test, group_train, group_test, config)
    try:
        if metric_name == 'FPR':
            return FPR(*metric_params)
        elif metric_name == 'max_parity_ratio':
            return max_parity_ratio(*metric_params)
        elif metric_name == 'equalized_odds_diff':
            return equalized_odds(*metric_params)
        elif metric_name == 'selection_rate':
            return positive_predictions(*metric_params)
        elif metric_name == 'accuracy':
            return accuracy(*metric_params)
        elif metric_name == 'calibration_score':
            calibration_score(*metric_params)
    except Exception as e:
        # just to catch metric errors -- don't want one application to mess up a whole run
        print(metric_name, str(mdl_obj), config['dataset_name'])
        print(e)
