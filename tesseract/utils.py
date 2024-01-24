# -*- coding: utf-8 -*-

"""
utils.py
~~~~~~~~

A selection of useful helper functions used throughout the Tesseract library.

"""

import logging
from datetime import datetime, date
from functools import wraps
from timeit import default_timer as timer

import numpy as np


def resolve_date(d):
    """Convert a str or date to an appropriate datetime.

    Strings should be of the format '%Y', '%Y-%m or '%Y-%m-%d', for example:
    '2012', '1994-02' or '1991-12-11'. Date objects with no time information
    will be rounded down to the midnight beginning that date.

    Args:
        d (Union[str, date]): The string or date to convert.

    Returns:
        datetime: The parsed datetime equivalent of d.
    """
    if isinstance(d, datetime):
        return d

    if isinstance(d, date):
        return datetime.combine(d, datetime.min.time())

    for fmt in ('%Y', '%Y-%m', '%Y-%m-%d'):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            pass

    raise ValueError('date string format not recognized.')


def check_for_raw_scores(y_pred):
    # Heuristic to check if input are raw scores
    if y_pred.ndim > 1:
        for v in y_pred:
            if ((np.linalg.norm(v, 0),
                 np.linalg.norm(v), 2) != (1, 1)):
                return True
    return False


def select_prediction_function(clf, scores_only=False, labels_only=False):
    if hasattr(clf, 'predict_proba') and not labels_only:
        prediction_function = clf.predict_proba
    elif hasattr(clf, 'decision_function') and not labels_only:
        prediction_function = clf.decision_function
    elif hasattr(clf, 'predict') and not scores_only:
        prediction_function = clf.predict
    else:
        raise TypeError(
            'Unsure how to handle predictions with '
            'classifier of type {}.'.format(clf.__class__))
    return prediction_function


def resolve_categorical(y):
    return np.argmax(y, 1) if y.ndim > 1 else y


def binary_labels(array, positive='malicious', negative='benign'):
    return [positive if x else negative for x in array]


def parse_percentage(n):
    return float(n[:-1]) / 100


def resolve_percentage(n):
    return parse_percentage(n) if isinstance(n, str) else n


def seconds_to_time(seconds):
    """Return a nicely formatted time given the number of seconds."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "%d days, %02d hours, %02d minutes, %02d seconds" % (d, h, m, s)


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start = timer()
        result = f(*args, **kwargs)
        elapsed = seconds_to_time(timer() - start)
        logging.debug('{} took: {}'.format(f.__name__, elapsed))
        return result

    return wrap
