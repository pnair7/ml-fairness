from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def logistic_regression(X_train, X_test, y_train, y_test, group_train, group_test, config, class_weight = 'balanced'):
    mdl = LogisticRegression(class_weight = class_weight, solver = 'sag', max_iter = 1000, random_state = 42)
    mdl.fit(X_train, y_train)
    results = list(mdl.predict(X_test))
    return results, mdl

def decision_tree(X_train, X_test, y_train, y_test, group_train, group_test, config):
    clf = DecisionTreeClassifier(random_state = 42)
    clf.fit(X_train, y_train)
    results = list(clf.predict(X_test))
    return results, clf
