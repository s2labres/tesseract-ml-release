# -*- coding: utf-8 -*-

"""
selection.py
~~~~~~~~~~~~

# TODO | Add module description

"""

import numpy as np

from tesseract import utils
from tesseract.evaluation import TrackingStage


class Selector(TrackingStage):
    def __init__(self, schedule=1, tracking=True, interaction='intersection'):
        super().__init__(schedule, tracking, interaction)
        self.selection_history = []

    def query_wrapper(self, clf, X_train, y_train, t_train,
                      X_test, y_test, t_test, previously_selected):
        # Pass parameters straight through to query implementation
        selected = self.query(clf, X_train, y_train, t_train,
                              X_test, y_test, t_test, previously_selected)

        if self.tracking:
            self.selection_history.append(selected)

        # Merge results with those of previous selectors
        selected = self.merge_results(previously_selected, selected)

        return np.array(selected)

    def query(self, clf, X_train, y_train, t_train,
              X_test, y_test, t_test, previously_selected):
        raise NotImplementedError('Selector must be subclassed')


class FullRetrainingSelector(Selector):
    def __init__(self, schedule=1, tracking=True, interaction='intersection'):
        super().__init__(schedule, tracking, interaction)

    def query(self, clf, X_train, y_train, t_train,
              X_test, y_test, t_test, previously_selected):
        return range(len(y_test))


class ActiveLearningSelector(Selector):
    def __init__(self, n, query_strategy, schedule=1,
                 tracking=True, interaction='intersection'):
        super().__init__(schedule, tracking, interaction)
        self.n = n
        self.query_strategy = query_strategy

    def query(self, clf, X_train, y_train, t_train,
              X_test, y_test, t_test, previously_selected):
        # Parse percentage if string passed in as n (eg. '20%')
        m = int(utils.parse_percentage(self.n) * len(y_test)
                if isinstance(self.n, str) else self.n)
        return self.query_strategy(clf, X_train, y_train, t_train,
                                   X_test, y_test, t_test,
                                   previously_selected, m)


class UncertaintySamplingSelector(Selector):
    def __init__(self, n, schedule=1, tracking=True,
                 interaction='intersection'):
        super().__init__(schedule, tracking, interaction)
        self.n = n

    def query(self, clf, X_train, y_train, t_train,
              X_test, y_test, t_test, previously_selected):
        # Parse percentage if string passed in as n (eg. '20%')
        m = int(utils.parse_percentage(self.n) * len(y_test)
                if isinstance(self.n, str) else self.n)

        # e.g. clf is a RandomForestsClassifier or SVC(probability=True)
        if hasattr(clf, 'predict_proba'):
            y_probs = clf.predict_proba(X_test)
            selected_indexes = probabilistic_uncertainty(y_probs, m)

        # e.g. clf is a LinearSVC or SVC
        elif hasattr(clf, 'decision_function'):
            y_raw = clf.decision_function(X_test)
            selected_indexes = closest_to_hyperplane(y_raw, m)

        else:
            raise TypeError(
                'Unsure how to handle uncertainty sampling with '
                'classifier of type {}.'.format(clf.__class__))

        return selected_indexes


def closest_to_hyperplane(distances, n):
    """Perform uncertainty sampling using distance from the hyperplane.

    Uncertainty sampling with SVMs is equivalent to selecting the samples
    closest to the decision boundary (hyperplane in binary classification).

    This is shown by Tong and Koller [ICML 2000]:
    https://dl.acm.org/citation.cfm?id=944793

    The intuition is also well explained by Kremer, Pederson, Igel [WIREs 2014]:
    http://image.diku.dk/jank/papers/WIREs2014.pdf

    The process for selecting the objects is as follows:

        1. Consider only absolute distances.
        2. Argsort from least distance to greatest.
        3. Take the n smallest (closest to the hyperplane).

    Args:
        distances: The list of distances to use as metrics.
        n: The number of samples to mark as 'most uncertain'.

    Returns:
        list: The indexes corresponding to the 'most uncertain' samples.

    """
    absolute = np.abs(distances)
    indexes = np.argsort(absolute)
    return indexes[:n]


def probabilistic_uncertainty(probs, n):
    """Perform uncertainty sampling using least confidence.

    An excellent discussion of active learning strategies including a
    comparison of three different uncertainty measures: least confidence,
    margin sampling and entropy (all of which are equivalent in binary
    classification) can be found in Burr Settles' literature review:

    http://burrsettles.com/pub/settles.activelearning.pdf

    The process for selecting the objects is as follows:

        1. Consider 'uncertainty' only (1 - the highest class probability).
        2. Argsort and reverse to sort from least to most certain.
        3. Take the n smallest (most uncertain).

    Args:
        probs: The list of probabilities to use as metrics.
        n: The number of samples to mark as 'most uncertain'.

    Returns:
        list: The indexes corresponding to the 'most uncertain' samples.

    """
    uncertainty = np.array([1 - np.max(x) for x in probs])
    indexes = np.argsort(uncertainty)[::-1]
    return indexes[:n]
