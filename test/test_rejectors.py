# -*- coding: utf-8 -*-

"""
test_rejectors.py
~~~~~~~~~~~~~~~~~

Unit tests for testing rejectors.py.

"""

import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

from tesseract import temporal, mock, metrics, rejection, evaluation
from tesseract.rejection import ThresholdRejector


class TestRejectors(unittest.TestCase):
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

        self.svm = SVC(kernel='linear', probability=False)
        self.svm.fit(X_train, y_train)

        self.rf = RandomForestClassifier(n_estimators=101, max_depth=64)
        self.rf.fit(X_train, y_train)

    def test_threshold_rejector_rf(self):
        t_rejector = ThresholdRejector('<', 0.9)
        results = evaluation.fit_predict_update(
            self.rf, self.X_train, self.X_tests,
            self.y_train, self.y_tests,
            self.t_train, self.t_tests,
            rejectors=[t_rejector])

        metrics.print_metrics(results)

        # Check that something was rejected each period,
        # more thorough tests are certainly desired!

        for i in range(len(self.y_tests)):
            self.assertGreater(results['rejected'][i], 0)

    def test_threshold_rejector_svm_between(self):
        t_rejector = ThresholdRejector('><', (-5, 5))
        results = evaluation.fit_predict_update(
            self.svm, self.X_train, self.X_tests,
            self.y_train, self.y_tests,
            self.t_train, self.t_tests,
            rejectors=[t_rejector])

        metrics.print_metrics(results)

        for i in range(len(self.y_tests)):
            self.assertGreater(results['rejected'][i], 0)

    def test_threshold_rejector_svm_outside(self):
        t_rejector = ThresholdRejector('<>', (-5, 5))
        results = evaluation.fit_predict_update(
            self.svm, self.X_train, self.X_tests,
            self.y_train, self.y_tests,
            self.t_train, self.t_tests,
            rejectors=[t_rejector])

        metrics.print_metrics(results)

        for i in range(len(self.y_tests)):
            self.assertGreater(results['rejected'][i], 0)


class TestRejection(unittest.TestCase):
    def test_thresholds(self):
        # Test partitions of 1 year
        X, y, t = mock.generate_binary_test_data(10000, '2014', end='2016')
        splits = temporal.time_aware_train_test_split(
            X, y, t, 12, 1, granularity='month', start_date='2014')
        X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

        clf = LinearSVC()
        aa = rejection.alpha_assessment(clf, X_train, y_train, folds=5)
        n_quartiles, p_quartiles = rejection.quartiles(aa)
        n_threshold, p_threshold = n_quartiles[3], p_quartiles[1]
        print(n_threshold, p_threshold)

        rejection_options = {'thresholds': [n_threshold, p_threshold],
                             'comparators': ['<', '>']}
