# -*- coding: utf-8 -*-

"""
test_selectors.py
~~~~~~~~~~~~~~~~~

Unit tests for testing selectors.py.

"""

import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from tesseract import temporal, mock, metrics, evaluation
from tesseract.selection import FullRetrainingSelector, ActiveLearningSelector, \
    UncertaintySamplingSelector


class TestSelectors(unittest.TestCase):
    def setUp(self):
        # Test partitions of 1 year
        X, y, t = mock.generate_binary_test_data(10000, '2020')

        splits = temporal.time_aware_train_test_split(
            X, y, t, train_size=6, test_size=2, granularity='month')
        X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

        self.X_train = X_train
        self.y_train = y_train
        self.X_tests = X_tests
        self.y_tests = y_tests
        self.t_train = t_train
        self.t_tests = t_tests

        self.svm = LinearSVC()
        self.svm.fit(X_train, y_train)

        self.rf = RandomForestClassifier(n_estimators=101, max_depth=64)
        self.rf.fit(X_train, y_train)

    def test_full_retraining(self):
        for clf in (self.svm, self.rf):
            results = evaluation.fit_predict_update(
                clf, self.X_train, self.X_tests,
                self.y_train, self.y_tests,
                self.t_train, self.t_tests,
                selectors=[FullRetrainingSelector()])

            metrics.print_metrics(results)

            for i in range(1, len(self.y_tests)):
                expected = results['train_tot'][i - 1] + results['tot'][i - 1]
                actual = results['train_tot'][i]

                self.assertEqual(expected, actual)

    def test_active_learning(self):
        def closest_to_hyperplane(*args):
            clf, X_test, n = args[0], args[4], args[-1]
            y_raw = clf.decision_function(X_test)
            absolute = np.abs(y_raw)
            indexes = np.argsort(absolute)
            return indexes[:n]

        results = evaluation.fit_predict_update(
            self.svm, self.X_train, self.X_tests,
            self.y_train, self.y_tests,
            self.t_train, self.t_tests,
            selectors=[ActiveLearningSelector(
                '20%', closest_to_hyperplane)])

        metrics.print_metrics(results)

        for i in range(1, len(self.y_tests)):
            expected = int(results['train_tot'][i - 1] +
                           results['tot'][i - 1] * 0.2)
            actual = results['train_tot'][i]

            self.assertEqual(expected, actual)

    def test_uncertainty_sampling(self):
        for clf in (self.svm, self.rf):
            results = evaluation.fit_predict_update(
                clf, self.X_train, self.X_tests,
                self.y_train, self.y_tests,
                self.t_train, self.t_tests,
                selectors=[UncertaintySamplingSelector('20%')])

            metrics.print_metrics(results)

            for i in range(1, len(self.y_tests)):
                expected = int(results['train_tot'][i - 1] +
                               results['tot'][i - 1] * 0.2)
                actual = results['train_tot'][i]

                self.assertEqual(expected, actual)
