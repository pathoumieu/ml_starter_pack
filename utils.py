import numpy as np
import pandas as pd
from numba import jit

from sklearn.preprocessing import LabelEncoder


def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage of dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
    verbose : bool, optional
        Print memory reduction if True, by default True

    Returns
    -------
    pandas.DataFrame
        Reduced DataFrame.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)\n'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_csv_sample(csv_path, sample_frac=1.0):
    """Load reduced sample dataframe from csv.

    Parameters
    ----------
    csv_path : str
        Path to csv.
    sample_frac : float, optional
        Sample size, by default 1.0

    Returns
    -------
    pandas.DataFrame
        Reduced DataFrame.
    """
    try:
        assert (sample_frac >= 0.0) and (sample_frac <= 1.0)
    except AssertionError:
        raise ValueError('Sample fraction is not in [0, 1]')

    if sample_frac == 1.0:
        return reduce_mem_usage(pd.read_csv(csv_path))
    else:
        with open(csv_path) as f:
            n_lines = sum(1 for line in f)
        n_sample = int(n_lines * sample_frac)
        print(f"""There are {n_lines} in this file.
Sampling {n_sample} lines...
        """)
        rand_skip = np.sort(np.random.permutation(range(1, n_lines - n_sample)))
        return reduce_mem_usage(pd.read_csv(csv_path, skiprows=rand_skip))


def sample_hyperspace(hyperspace, n_sample=1):
    """
    Create sample of size n_sample of given hyperspace.

    Parameters
    ----------
    hyperspace : dict
        Dictionnary of type {'hp1': [..], 'hp2': [..]}.
    n_sample : int, optional
        Sample size, by default 1.

    Returns
    -------
    list
        List of dictionnaries representing random grid.
    """
    params_list = []
    for i in range(n_sample):
        params = {}
        for key, hp_list in hyperspace.items():
            hp = np.random.choice(hp_list)
            params[key] = hp
        if params in params_list:
            continue
        else:
            params_list.append(params)
    return params_list


def label_encode(X, X_test, column):
    """Label encode column in train and test datasets

    Parameters
    ----------
    X : pandas.DataFrame
        Train set.
    X_test : pandas.DataFrame
        Test set.
    column : str
        Column to label encode.

    Returns
    -------
    X : pandas.DataFrame
        Label encoded train set.
    X_test : pandas.DataFrame
        Label encoded test set.
    """
    le = LabelEncoder()

    X[column] = le.fit_transform(X[column])
    X_test[column] = le.transform(X_test[column])

    return X, X_test


@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc
