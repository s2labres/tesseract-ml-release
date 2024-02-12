import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import numpy
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction import DictVectorizer


def load_features(fname, shas=False):
    """Load feature set.

    Args:
        fname (str): The common prefix for the dataset.
            (e.g., 'data/features/drebin' -> 'data/features/drebin-[X|Y|meta].json')

        shas (bool): Whether to include shas. In some versions of the dataset,
            shas were included to double-check alignment - these are _not_ features
            and _must_ be removed before training.

    Returns:
        Tuple[List[Dict], List, List]: The features, labels, and timestamps
            for the dataset.

    """
    time_index = {}

    feature_path = os.path.join(os.path.dirname(fname), 'extended-features-{}.json')

    with open(feature_path.format("X"), 'rb') as f:
        X = json.load(f)
    with open(feature_path.format("y"), 'r') as f:
        y = json.load(f)

    with open(feature_path.format("meta"), 'r') as f:
        T = json.load(f)
        T = [o['dex_date'] for o in T]
        T = numpy.array([datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') if "T" in o
                         else datetime.strptime(o, '%Y-%m-%d %H:%M:%S') for o in T])

    vec = DictVectorizer()
    X = vec.fit_transform(X)
    y = numpy.asarray(y)

    for i in range(len(T)):
        t = T[i]
        if t.year not in time_index:
            time_index[t.year] = {}
        if t.month not in time_index[t.year]:
            time_index[t.year][t.month] = []
        time_index[t.year][t.month].append(i)

    return X, y, T, time_index


def load_range_dataset_w_benign(data_name, start_month, end_month, folder='data/'):
    if start_month != end_month:
        dataset_name = f'{start_month}to{end_month}'
    else:
        dataset_name = f'{start_month}'
    saved_data_file = os.path.join(folder, data_name, f'{dataset_name}_selected.npz')
    data = np.load(saved_data_file, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    y_mal_family = data['y_mal_family']
    return X_train, y_train, y_mal_family


def feature_reduce(clf, dim):
    if hasattr(clf, 'coef_'):
        select_index = np.argpartition(abs(clf.coef_[0]), -dim)[-dim:]
        return select_index
    else:
        print('Wrong classifier')
        exit(-1)


def fetch_data(path, subset, year):
    kept_mask = []
    X, y, t = read_vectorized_features(path, subset)
    for i in t:
        kept_mask.append(True if i.year == year else False)
    return X[kept_mask], y[kept_mask], t[kept_mask]


def read_vectorized_features(data_dir, subset, feature_version=2):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    extractor = PEFeatureExtractor(feature_version)
    ndim = extractor.dim

    X_path = os.path.join(data_dir, "X_{}.dat".format(subset))
    y_path = os.path.join(data_dir, "y_{}.dat".format(subset))
    y = np.memmap(y_path, dtype=np.float32, mode="r")
    train_rows = (y != -1)
    N = y.shape[0]
    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(N, ndim))
    # X = None
    # y = None

    t_dframe = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
    t_list = t_dframe['appeared'].to_list()
    re = {}
    for i in t_list:
        if i not in re:
            re[i] = 1
        else:
            re[i] += 1
    print(re)

    t = np.array([datetime.strptime(date, '%Y-%m').date() for date in t_list])
    return X[train_rows], y[train_rows], t[train_rows]


def load_dates(infile):
    """
    Parses infile for any dates formatted as YYYY/MM/DD, at most one
    per line. Returns a list of datetime.date objects, in order of
    encounter.
    """
    datere = re.compile(r'\d{4}/\d{2}/\d{2}')
    dates = []
    for line in open(infile, 'r', encoding='utf-8'):
        match = re.search(datere, line)
        if match:
            dates.append(datetime(*(map(int, match.group().split('/')))))
    return dates
