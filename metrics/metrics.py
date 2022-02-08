from fairlearn.metrics import MetricFrame, false_positive_rate, demographic_parity_ratio, equalized_odds_ratio, selection_rate, equalized_odds_difference
import numpy as np


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
