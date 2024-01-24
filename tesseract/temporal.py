# -*- coding: utf-8 -*-

"""
temporal.py
~~~~~~~~~~~

A module for working with and running time-aware evaluations. Most of the
functionality of this module falls into one of two categories: working with
arrays of datetimes or datetime-aligned series of data, and aggregating the
steps of the ML pipeline needed to conduct sound, time-aware evaluations.

"""
import bisect
import operator

import numpy as np
from dateutil.relativedelta import relativedelta

import tesseract.utils as utils


def assert_train_test_temporal_consistency(t_train, t_test):
    """Helper function to assert train-test temporal constraint (C1).

    All objects in the training set need to be temporally anterior to all
    objects in the testing set. Violating this constraint will positively bias
    the results by integrating "future" knowledge into the classifier.

    Args:
        t_train: An array of datetimes corresponding to the training set.
        t_test: An array of datetime corresponding to the testing set.

    Returns:
        bool: False if the partitioned dataset does _not_ adhere to C1,
            True otherwise.

    """
    for train_date in t_train:
        for test_date in t_test:
            if train_date > test_date:
                return False
    return True


def assert_positive_negative_temporal_consistency(y, t, month_variance=1):
    """Helper function to assert malware-goodware temporal constraint (C2).

    In any given testing period, all testing objects must be from the time
    window under test. In the malware domain this constraint has often been
    violated so that malware and goodware come from different time periods.

    If this is the case, it becomes impossible to tell whether a
    high-performing classifier is discriminating between malicious and benign
    objects or between old and new applications.

    Args:
        y: An array of ground-truth labels for each observation.
        t: An array of datetimes for each observation (aligned with y).
        month_variance: All malware and goodware should be between this many
            months.

    Returns:
        bool: False if the malware and goodware do not adhere to C2,
            True otherwise

    """
    positive = np.where(y == 1)[0]
    negative = np.where(y != 1)[0]
    positive_dates = t[positive]
    negative_dates = t[negative]

    for pos_date in positive_dates:
        for neg_date in negative_dates:
            if month_difference(pos_date, neg_date) > month_variance:
                return False
    return True


def month_difference(d1, d2):
    """Get the difference in months between two datetimes."""
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def time_aware_train_test_split(X, y, t, train_size, test_size,
                                granularity, start_date=None):
    """Partition a dataset composed of time-labelled objects.

    Args:
        X (np.ndarray, csr_matrix): Multi-dimensional array of predictors.
        y (np.ndarray): Array of output labels.
        t (np.ndarray): Array of timestamp tags.
        train_size (int): The training window size W (in τ).
        test_size (int): The testing window size Δ (in τ).
        granularity (str): The unit of time τ, used to denote the window size.
            Acceptable values are 'year|quarter|month|week|day'.
        start_date (date): The date to begin partioning from (eg. to align with
            the start of the year).

    Returns:
        (np.ndarray, list, np.ndarray, list, np.ndarray, list):
            Training partition of predictors X.
            List of testing partitions of predictors X.
            Training partition of output variables y.
            List of testing partitions of predictors y.
            Training partition of meta t.
            List of testing partitions of meta t.

    """
    # Get partitioned indexes
    train, tests = time_aware_indexes(t, train_size, test_size,
                                      granularity, start_date)

    # Partition predictors and labels
    X_actual, y_actual, t_actual = X[train], y[train], t[train]

    X_tests = [X[index_set] for index_set in tests]
    y_tests = [y[index_set] for index_set in tests]
    t_tests = [t[index_set] for index_set in tests]

    return X_actual, X_tests, y_actual, y_tests, t_actual, t_tests


def time_aware_indexes(t, train_size, test_size, granularity, start_date=None):
    """Return a list of indexes that partition the list t by time.

    Sorts the list of dates t before dividing into training and testing
    partitions, ensuring a 'history-aware' split in the ensuing classification
    task.


    Args:
        t (np.ndarray): Array of timestamp tags.
        train_size (int): The training window size W (in τ).
        test_size (int): The testing window size Δ (in τ).
        granularity (str): The unit of time τ, used to denote the window size.
            Acceptable values are 'year|quarter|month|week|day'.
        start_date (date): The date to begin partioning from (eg. to align with
            the start of the year).

    Returns:
        (list, list):
            Indexing for the training partition.
            List of indexings for the testing partitions.

    """
    # Order the dates as well as their original positions
    with_indexes = zip(t, range(len(t)))
    ordered = sorted(with_indexes, key=operator.itemgetter(0))

    # Split out the dates from the indexes
    dates = [tup[0] for tup in ordered]
    indexes = [tup[1] for tup in ordered]

    # Get earliest date
    start_date = utils.resolve_date(start_date) if start_date else ordered[0][0]

    # Slice out training partition
    boundary = start_date + get_relative_delta(train_size, granularity)
    to_idx = bisect.bisect_left(dates, boundary)
    train = indexes[:to_idx]

    tests = []
    # Slice out testing partitions
    while to_idx < len(indexes):
        boundary += get_relative_delta(test_size, granularity)
        from_idx = to_idx
        to_idx = bisect.bisect_left(dates, boundary)
        tests.append(indexes[from_idx:to_idx])

    return train, tests


def time_aware_partition(t, proportion):
    """Partition an array of dates based on the given proportion.

    The set of timestamps will be bisected with the left bisection sized by
    the given proportion.

    Args:
        t: An array of datetimes.
        proportion: The proportion by which to split the array.

    Returns:
        tuple: The two bisections of the array.
    """
    # Order the dates as well as their original positions
    indexes = np.argsort(t)

    # Divide ordered set in two
    boundary = int(proportion * len(indexes))

    return indexes[:boundary], indexes[boundary:]


def temporal_slice(X, y, t):
    raise NotImplementedError


def get_relative_delta(offset, granularity):
    """Get delta of size 'granularity'.

    Args:
        offset: The number of time units to offset by.
        granularity: The unit of time to offset by, expects one of
            'year', 'quarter', 'month', 'week', 'day'.

    Returns:
        The timedelta equivalent to offset * granularity.

    """
    # Make allowances for year(s), quarter(s), month(s), week(s), day(s)
    granularity = granularity[:-1] if granularity[-1] == 's' else granularity
    try:
        return {
            'year': relativedelta(years=offset),
            'quarter': relativedelta(months=3 * offset),
            'month': relativedelta(months=offset),
            'week': relativedelta(weeks=offset),
            'day': relativedelta(days=offset),
        }[granularity]
    except KeyError:
        raise ValueError('granularity not recognised, try: '
                         'year|quarter|month|week|day')
