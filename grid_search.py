# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

from train_and_test import get_all_data, train_path, test_path

X_train, Y_train, X_test, Y_test, X_train_norm, Y_train_norm, X_test_norm, Y_test_norm = get_all_data(train_path,
                                                                                                      test_path)


def run_search(model, param_grid, n_jobs=1, norm=True):
    """
    Runs grid search on a given model with the selected param_grid, prints out train and test accuracy.

    Args:
        model: estimator object, using scikit-learn estimator interface
        param_grid: dict, dictionary of parameters (see: GridSearchCV's documentation)
        n_jobs: int

    Returns:
        search: GridSearchCV object
    """

    search = GridSearchCV(model, param_grid, n_jobs=n_jobs, verbose=1)

    if norm:
        search.fit(X_train_norm, Y_train_norm)

        print("Train score:", search.score(X_train_norm, Y_train_norm))
        print("Test score:", search.score(X_test_norm, Y_test_norm))

        print("Best params:", search.best_params_)
    else:
        search.fit(X_train, Y_train)

        print("Train score:", search.score(X_train, Y_train))
        print("Test score:", search.score(X_test, Y_test))

        print("Best params:", search.best_params_)

    return search
