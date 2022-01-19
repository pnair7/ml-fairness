import pandas as pd 
import numpy as np

def split_data(X, y, groups, train_size = 0.75):
    '''
    X and y are Pandas DataFrames of features and labels, respectively. (See ~/main.py).
    groups is a Series of the group column.
    Requires all three to share a common index -- which they should, since they're generated from
    the same source DataFrame. But if this breaks in the future, maybe that's why.
    '''
    assert len(X) == len(y) == len(groups)

    random.seed(42)
    X_shuffled = X.sample(frac = 1)
    y_shuffled = y.loc[X_shuffled]
    groups = y.loc[X_shuffled]

    # generate train, test, groups split by train_size
    n_train = int(train_size * len(X))
    X_train = X_shuffled.iloc[:n_train]
    X_test = X_shuffled.iloc[n_train:]
    y_train = y_shuffled.iloc[:n_train]
    y_test = y_shuffled.iloc[n_train:]
    group_train = groups.iloc[:n_train]
    group_test = groups.iloc[n_train:]

    return X_train, X_test, y_train, y_test, group_train, group_test