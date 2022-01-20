from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, X_test, y_train, y_test, group_train, group_test, config, class_weight = None):
    mdl = LogisticRegression(class_weight = class_weight, solver = 'sag', max_iter = 1000)
    mdl.fit(X_train, y_train)
    results = mdl.predict(X_test)
    return results, mdl