#!/bin/python3.5

import argparse

from train_and_test import get_all_data, train_and_validate, train_path, test_path
from settings import model_params

if __name__ == '__main__':
    # X_train, Y_train, X_test, Y_test, X_train_norm, Y_train_norm, X_test_norm, Y_test_norm = get_all_data(train_path,
    #                                                                                                      test_path)
    # supported algorithms:
    # train_and_validate()

    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', help='Name of the algorithm to run. One of: naive-bayes,\
     naive-bayes-g, naive-bayes-m, naive-bayes-b, decision-tree, knn, svm, rf, et')

    parser.add_argument('-n', '--normalize', help='Normalize data', action='store_true')
    # parser.add_argument('-s', '--settings', help='Use model parameters stored in settings.py', action='store_true')
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