# This file will contain your data_preprocessing.py script.
import numpy as np
import pandas as pd


def read_data(path):
    """
    Convenience function for reading the raw data into a DataFrame.
    
    Args:
        path: str, path to the data file (csv format)
    
    Returns:
        DataFrame containing the raw data

    """

    column_names = ['age', 'workclass',
                    'fnlwgt', 'education',
                    'education-num', 'marital-status',
                    'occupation', 'relationship',
                    'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week',
                    'native-country', 'income']

    data = pd.read_csv(path, header=None)
    data.columns = column_names

    return data


def fix_labels(data, inplace=False):
    """DEPRECATED - use categorical_to_bool instead (keeping it just in case)
    Converts <=50K and >50K to 0 and 1 respectively.
    
    Args:
        data: DataFrame, containing the raw data
        inplace: boolean, whether to modify the original DataFrame
        
    Returns:
        DataFrame with fixed labels
    """

    if not inplace:
        data = data.copy()
    data['label'] = 0
    data.loc[data['income'] == ' >50K', 'label'] = 1
    del data['income']
    return data


def categorical_to_bool(data, column, to_one, new_name=None, inplace=False):
    """
    Converts a column of arbitrary datatype to a boolean column. 
    
    Args:
        data: DataFrame, containing the raw data
        column: str, name of the categorical column
        to_one: data.column.dtype, value supposed to be set to 1 (the other is set to 0)
        new_name: NoneType or str, name of the new column (==column if new_name is None)
        inplace: boolean, whether to modify the original DataFrame
    
    Returns:
        DataFrame with the chosen column changed to a boolean column
    """

    if not inplace:
        data = data.copy()

    if new_name is None:
        new_name = column

    temp_name = new_name + 'temp'
    assert temp_name not in data.columns

    data[temp_name] = 0
    data.loc[data[column] == to_one, temp_name] = 1
    del data[column]
    data.rename(columns={temp_name: new_name}, inplace=True)

    return data


def remove_nan(data, nan='?', inplace=False):
    """
    Removes rows with missing values.
    
    Args:
        data: DataFrame, containing the raw data
        nan: any, current placeholder for the missing values
        inplace: boolean, whether to modfiy the original DataFrame
    
    Returns:
        Cleaned DataFrame
    """

    if not inplace:
        data = data.copy()

    data.replace(nan, np.nan, inplace=True)
    data.dropna(axis=0, how='any', inplace=True)

    data.reset_index(inplace=True)
    del data['index']

    return data


def strip_data(data, inplace=False):
    """
    Removes leading and trailing whitespaces from the data.
    """

    if not inplace:
        data = data.copy()

    for col in data.columns:
        if data[col].dtype == 'O':
            data[col] = data[col].str.strip()

    return data


def categorical_to_onehot(data, column):
    """
    Converts a column of arbitrary datatype to a number of one-hot encoded columns.
    
    Args:
        data: DataFrame, containing the raw data
        column: str, name of the categorical column
        
    Returns:
        Modified DataFrame
    """

    data = data.join(pd.get_dummies(data[column]))

    del data[column]

    return data


def many_cat_to_onehot(data, columns):
    """
    Convenience function for converting a number of columns to their one-hot encoded versions.
    
    Args:
        data: DataFrame, containing the raw data
        columns: list of strings, containing names of columns to encode
    
    Returns:
        Modified DataFrame
    """

    for column in columns:
        data = categorical_to_onehot(data, column)

    return data


def prepare_data(path):
    """
    Produces a cleaned and preprocessed DataFrame containing the data, ready to plug into a learning algorithm.
    
    Args:
        path: str, path to the data file (csv format)
        
    Returns:
        DataFrame with preprocessed data
    """

    data = read_data(path)

    # Clean data
    strip_data(data, inplace=True)  # Remove whitespaces from string values
    remove_nan(data, nan='?', inplace=True)  # Remove rows with missing data
    del data['education']  # Remove redundant column

    # Fix binary features
    data = categorical_to_bool(data, column='income', to_one='>50K', new_name='label')
    data = categorical_to_bool(data, column='sex', to_one='Male')

    # Fix non-binary features
    data = many_cat_to_onehot(data, ['workclass', 'marital-status',
                                     'native-country', 'race',
                                     'relationship', 'occupation'])

    return data


def normalize_column(data, column, inplace=False):
    """
    Normalizes the chosen column to 0 mean and 1 variance.
    
    Args:
        data: DataFrame, containing the raw data
        column: str, name of the column to be normalized
        inplace: boolean, whether to modify the original DataFrame
        
    Returns:
        data: DataFrame with the chosen column normalized
        mean: float-like, mean of the column before normalizing
        std: float-like, std of the column before normalizing
    """

    if not inplace:
        data = data.copy()

    mean = data[column].mean()
    std = data[column].std()

    normalized_col = (data[column] - mean) / std

    data[column] = normalized_col

    return data, mean, std


def normalize_multiple_columns(data, columns, inplace=False):
    """
    Convenience function for normalizing several columns.
    
    Args:
        data: DataFrame, containing the raw data
        columns: list of strings, names of the columns to be normalized
        inplace: boolean, whether to modify the original DataFrame
    
    Returns:
        data: DataFrame
        means: dict
        stds: dict
    """

    if not inplace:
        data = data.copy()

    means = dict()
    stds = dict()
    for column in columns:
        data, mean, std = normalize_column(data, column)
        means[column] = mean
        stds[column] = std

    return data, means, stds


def normalize_test_data(data, means, stds, inplace=False):
    """
    Normalizes data with given means and stds.
    
    Args:
        data: DataFrame, containing test data
        means: dict, containing means of columns to normalize
        stds: dict, containing stds of columns to normalize
        inplace: boolean, whether to modify the original DataFrame
        
    Returns:
        data: DataFrame, containing normalized data
    """

    assert means.keys() == stds.keys()

    if not inplace:
        data = data.copy()

    columns = means.keys()

    for column in columns:
        data[column] = (data[column] - means[column]) / stds[column]

    return data


def split_data(data, label_name='label'):
    """
    Splits the DataFrame into two: one containing the features, the other containing labels.
    
    Args:
        data: DataFrame, containing the raw data
        label_name: str, name of the column containing the labels
        
    Returns:
        data: DataFrame, containing features
        labels: DataFrame, containing labels
    """

    data = data.copy()
    labels = data[[label_name]]

    del data[label_name]
    return data, labels


def add_missing_cols(train_data, test_data, inplace=False):
    """
    Adds any columns present in train_data, that are absent in test_data.
    
    Args:
        train_data: DataFrame
        test_data: DataFrame
        inplace: boolean, whether to modify the original DataFrame
        
    Returns:
        test_data: DataFrame with added columns
    """

    if not inplace:
        test_data = test_data.copy()

    missing_cols = set(train_data.columns) - set(test_data.columns)

    for col in missing_cols:
        test_data[col] = 0

    test_data = test_data[train_data.columns]

    return test_data
