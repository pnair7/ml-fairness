from fairlearn.metrics import MetricFrame, false_positive_rate, demographic_parity_ratio, equalized_odds_ratio, selection_rate, equalized_odds_difference
import numpy as np
import pandas as pd
# from sklearn.calibration import CalibrationDisplay
from sklearn import metrics
import matplotlib.pyplot as plt


def FPR(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    fpr_metric = MetricFrame(
        metrics=false_positive_rate, y_true=y_true, y_pred=y_pred,
        sensitive_features=groups)
    return fpr_metric.difference(method='between_groups')


def f1_score_range(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred, 'groups': groups})
    f1_vals = []
    for group in np.unique(groups):
        subset = df[df['groups'] == group]
        f1_vals.append(metrics.f1_score(subset['y_true'], subset['y_pred']))
    return np.ptp(f1_vals)


def recall_range(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred, 'groups': groups})
    recall_vals = []
    for group in np.unique(groups):
        subset = df[df['groups'] == group]
        recall_vals.append(metrics.recall_score(subset['y_true'], subset['y_pred']))
    return np.ptp(recall_vals)


def accuracy_range(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred = list(results)
    groups = list(group_test)

    df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred, 'groups': groups})
    recall_vals = []
    for group in np.unique(groups):
        subset = df[df['groups'] == group]
        recall_vals.append(metrics.accuracy_score(subset['y_true'], subset['y_pred']))
    return np.ptp(recall_vals)


def brier_score_range(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred_prob = list(np.array(results_prob)[:, 1])
    groups = list(group_test)

    df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred_prob, 'groups': groups})
    brier_vals = []
    for group in np.unique(groups):
        subset = df[df['groups'] == group]
        brier_vals.append(metrics.brier_score_loss(subset['y_true'], subset['y_pred']))
    return np.ptp(brier_vals)


def overall_brier_score(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(y_test)
    y_pred_prob = list(np.array(results_prob)[:, 1])
    groups = list(group_test)

    df = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred_prob, 'groups': groups})
    return metrics.brier_score_loss(df['y_true'], df['y_pred'])


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


def overall_accuracy(metric_name, results, results_prob, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
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
