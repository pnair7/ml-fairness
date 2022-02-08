from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def logistic_regression(X_train, X_test, y_train, y_test, group_train, group_test, config, class_weight='balanced'):
    mdl = LogisticRegression(class_weight=class_weight, solver='sag', random_state=42)
    mdl.fit(X_train, y_train)
    results = list(mdl.predict(X_test))
    results_prob = list(mdl.predict_proba(X_test))
    return results, results_prob, mdl


def decision_tree(X_train, X_test, y_train, y_test, group_train, group_test, config):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    results = list(clf.predict(X_test))
    results_prob = list(clf.predict_proba(X_test))
    return results, results_prob, clf


def random_forest(X_train, X_test, y_train, y_test, group_train, group_test, config):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    results = list(clf.predict(X_test))
    results_prob = list(clf.predict_proba(X_test))
    return results, results_prob, clf


def multilayer_perceptron(X_train, X_test, y_train, y_test, group_train, group_test, config):
    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_train, y_train)
    results = list(mlp.predict(X_test))
    results_prob = list(mlp.predict_proba(X_test))
    return results, results_prob, mlp
