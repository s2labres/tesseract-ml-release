# -*- coding: utf-8 -*-

"""
mock.py
~~~~~~~

A module for generating test distributions for use with Tesseract.

"""

from datetime import datetime

import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.datasets import make_classification

from tesseract.utils import resolve_date


def generate_binary_test_data(n_samples, start, end=None, random_state=None):
    """Generate a test dataset suitable for binary classification.

    Args:
        n_samples (int): The number of examples to create between start and end.
        start (str): The start date of the range to generate examples within.
        end (str): The end date of the range to generate examples within.
        random_state (int): A random number seed.

    Returns:
        np.ndarray: Array of two-dimensional predictors X.
        np.ndarray: Array of output variables y.
        np.ndarray: Array of datetimes for each example.

    """
    X, y = make_classification(n_samples, n_features=2, n_informative=2, n_redundant=0, class_sep=1.5,
                               random_state=random_state)
    t = generate_time_data(n_samples, start, end)
    return X, y, t


def generate_time_data(n_samples, start, end=None, random_state=None):
    """Randomly sample from the given date range.

    Args:
        n_samples (int): The number of dates to create between start and end.
        start (str): The start date of the range to sample from.
        end (str): The end date of the range to sample from.
        random_state (int): A random number seed.

    Returns:
        np.ndarray: Array of datetimes sampled within the given range.

    """
    start = resolve_date(start)
    end = resolve_date(end) if end else datetime(start.year, 12, 31)

    np.random.seed(random_state)
    delta = int((end - start).total_seconds())
    offsets = [np.random.randint(delta) for _ in range(n_samples)]
    return np.array([start + relativedelta(seconds=x) for x in offsets])
