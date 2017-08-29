model_params = {
    'naive-bayes': {},
    'naive-bayes-g': {'priors': [0.9, 0.1]},
    'naive-bayes-m': {'fit_prior': True, 'alpha': 0.01, 'class_prior': [0.8, 0.2]},
    'naive-bayes-b': {'binarize': 10, 'fit_prior': True, 'alpha': 0.01, 'class_prior': [0.8, 0.2]},
    'decision-tree': {'criterion': 'gini', 'max_features': None, 'min_impurity_decrease': .0001},
    'knn': {},
    'svm': {'dual': True, 'max_iter': 200},
    'rf': {'criterion': 'gini', 'n_estimators': 200, 'min_impurity_decrease': 0.0,
           'min_samples_split': 8, 'min_samples_leaf': 2},
    'et': {'criterion': 'gini', 'n_estimators': 200, 'min_impurity_decrease': 0.0,
           'min_samples_split': 8, 'min_samples_leaf': 2},
    'logreg': {'fit_intercept': True, 'penalty': 'l2', 'dual': False},
    'xgb': {}
}
