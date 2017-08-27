# This file will contain your train_and_test.py script.
# import numpy as np
# import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt

from data_preprocessing import prepare_data, normalize_multiple_columns, \
                        normalize_test_data, split_data, add_missing_cols

train_path = './data/train_data.txt'
test_path = './data/test_data.txt'
NUMERIC_COLUMNS = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


def get_train_data(path, normalize=True, num_cols=NUMERIC_COLUMNS):
    """
    Convenience function for extracting and optionally normalizing data.

    Args:
        path: str, path to the data file
        normalize: boolean, whether to normalize numeric columns
        num_cols: list-like, if normalize is True, list of columns to normalize

    Returns:
        train_data: DataFrame
        means: dict, containing means of normalized columns (None if normalize is False)
        stds: dict, containing stds of normalized columns (None if normalize if False)
    """

    train_data = prepare_data(path)
    means = stds = None

    if normalize:
        train_data, means, stds = normalize_multiple_columns(train_data, num_cols)

    return train_data, means, stds


def get_test_data(path, normalize=True, means=None, stds=None, train_data=None):
    """
    Convenience function for extracting and optionally normalizing test data.

    Args:
        path: str, path to the data file
        normalize: boolean, whether to normalize numeric columns
        means: dict, containing means of columns to be normalized
        stds: dict, containing stds of columns to be normalized
        train_data: DataFrame, if not None, referrence DataFrame for adding missing columns to test_data

    Returns:
        test_data: DataFrame
    """

    if normalize:
        assert means is not None
        assert stds is not None

    test_data = prepare_data(path)

    if train_data is not None:
        test_data = add_missing_cols(train_data, test_data)
        # test_data = test_data[train_data.columns]

    if normalize:
        test_data = normalize_test_data(test_data, means, stds)

    return test_data


def get_feature_matrix(raw_data):
    """
    Convenience function converting a DataFrame to a feature matrix and a labels vector.

    Args:
        raw_data: DataFrame, containing raw data

    Returns:
        data: np.ndarray, feature matrix
        labels: np.ndarray, labels vector
    """

    data, labels = split_data(raw_data)
    data, labels = data.values, labels.values.ravel()

    return data, labels


def get_all_data(path_train, path_test, kind='both'):
    """
    Convenience function for extracting unnormalized and normalized feature matrices and labels.

    Args:
        path_train: str, path to the train data
        path_test: str, path to the test data
        kind: str, one of 'both', 'norm', 'unnorm',
                    indicates whether to return normalized, unnormalized or both datasets

    Returns:
        X_train, Y_train, X_test, Y_test: np.ndarrays, feature matrices and labels
        X_train_norm, Y_train_norm, X_test_norm, Y_test_norm: np.ndarrays, normalized feature matrices and labels
    """
    train_data, means, stds = get_train_data(path_train, normalize=False)
    test_data = get_test_data(path_test, normalize=False, means=means, stds=stds, train_data=train_data)

    X_train, Y_train = get_feature_matrix(train_data)
    X_test, Y_test = get_feature_matrix(test_data)

    train_data_norm, means_norm, stds_norm = get_train_data(path_train, normalize=True)

    test_data_norm = get_test_data(path_test,
                                   normalize=True,
                                   means=means_norm,
                                   stds=stds_norm,
                                   train_data=train_data_norm)

    X_train_norm, Y_train_norm = get_feature_matrix(train_data_norm)
    X_test_norm, Y_test_norm = get_feature_matrix(test_data_norm)
    if kind == 'both':
        return X_train, Y_train, X_test, Y_test, X_train_norm, Y_train_norm, X_test_norm, Y_test_norm
    elif kind == 'norm':
        return X_train_norm, Y_train_norm, X_test_norm, Y_test_norm
    elif kind == 'unnorm':
        return X_train, Y_train, X_test, Y_test
    else:
        raise ValueError('Make sure the value of kind is one of \'both\', \'norm\', \'unnorm\'')


def train_and_validate(algorithm, X_train, Y_train, X_test, Y_test, suffix='', **params):
    """
    Trains the chosen algorithm on X_train and Y_trains, and evaluates it on train and test datasets.

    Args:
        algorithm: str, name of the algorithm
                   currently supported: naive-bayes, naive-bayes-g, naive-bayes-m, naive-bayes-b,
                                        decision-tree
        X_train: np.ndarray, training feature matrix
        Y_train: np.ndarray, training labels
        X_test: np.ndarray, test feature matrix
        Y_test: np.ndarray, test l  abels
        suffix: str, suffix to be added to the printed algorithm's name
        params: parameters to be passed to the algorithm

    Returns:
        train_score: float, accuracy on the training set
        test_score: float, accuracy on the test set
    """

    algo_dict = {'naive-bayes': GaussianNB,
                 'naive-bayes-g': GaussianNB,
                 'naive-bayes-m': MultinomialNB,
                 'naive-bayes-b': BernoulliNB,
                 'decision-tree': DecisionTreeClassifier,
                 'knn': KNeighborsClassifier,
                 'svm': LinearSVC,
                 'rf': RandomForestClassifier,
                 'et': ExtraTreesClassifier,
                 'logreg': LogisticRegression,
                 }

    assert algorithm in algo_dict

    model = algo_dict[algorithm]()
    if params:
        model.set_params(**params)

    model.fit(X_train, Y_train)
    train_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)

    print(algorithm + suffix, "Train accuracy:", round(train_score, 3))
    print(algorithm + suffix, "Test accuracy:", round(test_score, 3))

    return train_score, test_score
