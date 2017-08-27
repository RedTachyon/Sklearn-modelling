#!/bin/python3

import argparse

from train_and_test import get_all_data, train_and_validate, train_path, test_path
from settings import model_params


def main():
    """
    Runs an experiment with the model chosen as a CLI argument, with parameters specified in settings.py.
    Basically a wrapper for the train_and_validate function from train_and_test.py

    Example usage:
        ./eval_model svm -n
        ./eval_model rf

    Usage suggestions:
        -- Remember to use -n for svm.
           Actually, most models benefit from it, maybe except for decision tree-based models and naive-bayes models.
        -- naive-bayes-m does not accept -n at all.
        -- Change settings.py to set your own parameters.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', help='Name of the algorithm to run. One of: naive-bayes,\
     naive-bayes-g, naive-bayes-m, naive-bayes-b, decision-tree, knn, svm, rf, et, logreg')

    parser.add_argument('-n', '--normalize', help='Normalize data', action='store_true')
    parser.add_argument('-v', '--verbose', help='Give more verbose output', action='store_true')

    args = parser.parse_args()

    norm_kind = 'norm' if args.normalize else 'unnorm'
    X_train, Y_train, X_test, Y_test = get_all_data(path_train=train_path, path_test=test_path, kind=norm_kind)

    params = model_params[args.algorithm]

    if args.verbose:
        print('Model:', args.algorithm)
        print('Normalization:', str(args.normalize))
        print('Model parameters:', params)

    train_and_validate(args.algorithm, X_train, Y_train, X_test, Y_test, suffix='', **params)

if __name__ == '__main__':
    main()
