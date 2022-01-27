from fairlearn.metrics import MetricFrame, false_positive_rate


def FPR(metric_name, results, mdl_obj, X_train, X_test, y_train, y_test, group_train, group_test, config):
    y_true = list(results)
    y_pred = list(y_test)
    groups = list(group_test)

    fpr_metric = MetricFrame(
        metrics=false_positive_rate, y_true=y_true, y_pred=y_pred,
        sensitive_features=groups)
    return(fpr_metric.difference(method='between_groups'))
