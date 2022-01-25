from fairlearn.metrics import MetricFrame, false_positive_rate

def FPR(y_true, y_pred, groups):
    fpr_metric = MetricFrame(
        metrics=false_positive_rate, y_true=y_true, y_pred=y_pred, 
        sensitive_features=groups)
    print("difference in FPR = ", fpr_metric.difference(method='between_groups'))
    