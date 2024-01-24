# -*- coding: utf-8 -*-

"""
rejection.py
~~~~~~~~~~~~

# TODO | Add module description

"""
import logging
import os

import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from tqdm import tqdm

from tesseract import utils
from tesseract.evaluation import TrackingStage
from tesseract.temporal import time_aware_partition


class Rejector(TrackingStage):
    def __init__(self, schedule=1, tracking=True, interaction='intersection'):
        super().__init__(schedule, tracking, interaction)
        self.kept_history = []
        self.rejection_history = []

    def reject_wrapper(self, clf, X_train, y_train, t_train, X_test,
                       y_test, t_test, previously_kept, previously_rejected):
        # Pass parameters straight through to reject implementation
        kept, rejected = self.reject(clf, X_train, y_train, t_train,
                                     X_test, y_test, t_test, previously_kept,
                                     previously_rejected)

        if self.tracking:
            self.kept_history.append(kept)
            self.rejection_history.append(rejected)

        # Merge results with those of previous rejectors
        kept = self.merge_results(previously_kept, kept)
        rejected = self.merge_results(previously_rejected, rejected)

        return np.array(kept), np.array(rejected)

    def reject(self, clf, X_train, y_train, t_train,
               X_test, y_test, t_test, previously_kept, previously_rejected):
        raise NotImplementedError('Rejector must be subclassed')


class ThresholdRejector(Rejector):
    def __init__(self, operator, thresholds, point_score='credibility',
                 schedule=1, tracking=True, interaction='intersection'):
        super().__init__(schedule, tracking, interaction)

        self._single_threshold_ops = ('<', 'lesser',
                                      '>', 'greater')
        self._double_threshold_ops = ('<>', 'outside',
                                      '><', 'between')

        self._valid_operators = (self._single_threshold_ops +
                                 self._double_threshold_ops)

        self._valid_point_scores = ('credibility', 'confidence')

        self._check_params(
            operator, thresholds, point_score, tracking, interaction)

        self.point_score = point_score
        self.thresholds = thresholds
        self.operator = operator

        if hasattr(thresholds, '__len__'):
            self.threshold = max(thresholds)
            self.lower_threshold = min(thresholds)
        else:
            self.threshold = thresholds
            self.lower_threshold = None

    def reject(self, clf, X_train, y_train, t_train,
               X_test, y_test, t_test, previously_kept, previously_rejected):
        get_score = utils.select_prediction_function(clf)

        y_scores = get_score(X_test)

        # Resolve arrays where the scoring function outputs per-class scores

        if hasattr(y_scores[0], '__len__'):
            if self.point_score == 'credibility':
                # credibility = the highest score
                y_scores = np.array([max(v) for v in y_scores])
            elif self.point_score == 'confidence':
                # confidence = the highest score minus the next highest
                y_scores = np.array([max(v) - np.partition(v, -2)[-2]
                                     for v in y_scores])

        if self.operator in ('<', 'lesser'):
            rejected = np.where(y_scores < self.threshold)[0]

        elif self.operator in ('>', 'greater'):
            rejected = np.where(y_scores > self.threshold)[0]

        elif self.operator in ('<>', 'outside'):
            rejected = np.where(np.logical_or(y_scores < self.lower_threshold,
                                              y_scores > self.threshold))[0]
        elif self.operator in ('><', 'between'):
            rejected = np.where(np.logical_and(y_scores > self.lower_threshold,
                                               y_scores < self.threshold))[0]
        else:
            raise ValueError('Unrecognised comparator for rejection')
            # Add indexes that didn't pass to list of quarantined samples

        kept = np.setxor1d(rejected, np.arange(len(y_scores)))

        return kept, rejected

    def _check_params(self, operator, thresholds,
                      point_score, tracking, interaction):

        if hasattr(thresholds, '__len__') and len(thresholds) > 2:
            raise ValueError(
                'ThresholdRejector will only accept a '
                'maximum of 2 thresholds (one upper, one lower)')

        if operator not in self._valid_operators:
            raise ValueError(
                'Threshold comparison operator must be one of the '
                'following: {}'.format(self._valid_operators))

        if point_score not in self._valid_point_scores:
            raise ValueError(
                'Point scores must be one of the '
                'following: {}'.format(self._valid_point_scores))

        if (operator in self._double_threshold_ops and
                not hasattr(thresholds, '__len__')):
            raise ValueError('"{}" expects two thresholds'.format(operator))

        if (operator in self._single_threshold_ops and
                hasattr(thresholds, '__len__')):
            raise ValueError('"{}" expects a single threshold'.format(operator))


def quartiles(alpha_assessment_results, subkey='incorrect'):
    """Considering an alpha assessment, return the quartiles from the results.

    In well-separated alpha assessment results, quartiles can be useful for
    finding a good threshold (below which, predictions are discarded).

    Typically thresholds are Q3 of incorrect predictions and Q1 of correct
    predictions.

    Args:
        alpha_assessment_results: The results to derive quartiles from.
        subkey: 'correct' or 'incorrect'.

    Returns:
        tuple: The quartiles as they relate to the negative and positive class.

    """
    percentiles = [0, 25, 50, 75, 100]
    negative = alpha_assessment_results['negative_predictions'][subkey]
    positive = alpha_assessment_results['positive_predictions'][subkey]
    neg_quartiles = [np.percentile(negative, p) for p in percentiles]
    pos_quartiles = [np.percentile(positive, p) for p in percentiles]
    return neg_quartiles, pos_quartiles


def alpha_assessment(clf, X, y, folds=10):
    """Perform an alpha assessment on the given classifier and data.

    An alpha assessment is an assessment used in conformal evaluation to
    visually discern how separable the classifier's correct and incorrect
    prediction scores are.

    Highly separable scores allow the user to control a threshold below which
    they can designate predictions as being low-confidence, unreliable or even
    rejected. In the domain of malware classification, the rate at which a
    greater proportion of samples appear _below_ the threshold is indicative
    of the rate at which concept drift is occuring.

    A formal description and thorough evaluation of its uses is given in the
    Transcend paper by Jordaney et. al [USENIX 2017]:
    https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-jordaney.pdf

    Args:
        clf: The classifier to use to perform the assessment.
        X: An array of predictors.
        y: An array of output labels aligned with X.
        folds: The number of folds to perform during the K-fold.

    Returns:

    """
    if hasattr(clf, 'predict_proba'):
        f = 'predict_proba'
    elif hasattr(clf, 'decision_function'):
        f = 'decision_function'
    else:
        raise TypeError(
            'Unsure how to handle scoring with '
            'classifier of type {}.'.format(clf.__class__))

    # random_state was set to 22 however removed due to ValueError since shuffle=False
    cv = KFold(n_splits=folds, shuffle=False)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    y_score = cross_val_predict(clf, X, y, cv=cv, method=f)

    negative = np.where(y_pred == 0)[0]
    positive = np.where(y_pred == 1)[0]
    correct = np.where(y_pred == y)[0]
    incorrect = np.where(y_pred != y)[0]

    return {
        'negative_predictions': {
            'correct': y_score[np.intersect1d(negative, correct)],
            'incorrect': y_score[np.intersect1d(negative, incorrect)]},
        'positive_predictions': {
            'correct': y_score[np.intersect1d(positive, correct)],
            'incorrect': y_score[np.intersect1d(positive, incorrect)]}
    }
