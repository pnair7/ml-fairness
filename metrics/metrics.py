from fairlearn.metrics import MetricFrame, false_positive_rate, demographic_parity_ratio, equalized_odds_ratio, selection_rate, equalized_odds_difference
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt


def FPR(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    fpr_metric = MetricFrame(
        metrics=false_positive_rate, y_true=y_true, y_pred=y_pred,
        sensitive_features=groups)
    return fpr_metric.difference(method='between_groups')


def max_parity_ratio(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    return demographic_parity_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=groups)


def equalized_odds(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    return equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=groups)


def positive_predictions(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    return selection_rate(y_true=y_true, y_pred=y_pred)


def accuracy(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    return (pd.Series(y_true) == pd.Series(y_pred)).mean()


def calibration_score(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred_prob = list(np.array(results_prob)[:, 1])
    groups = list(group_test)

    df = pd.DataFrame.from_dict({
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'groups': groups
    })

    df['y_pred_decile'] = (10 * df['y_pred_prob']).astype(int) + 1
    disp = CalibrationDisplay.from_predictions(y_test, y_pred_prob, n_bins=10, strategy='quantile')
    plt.title(str(type(mdl_obj)))
    # plt.show() # Outputs calibration plots, if we need it

# TODO: counterfactual fairness
