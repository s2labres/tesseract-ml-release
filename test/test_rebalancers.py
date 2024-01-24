# -*- coding: utf-8 -*-

"""
test_rebalancers.py
~~~~~~~~~~~~~~~~~~~

Unit tests for testing rebalancers.py.

"""

import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tesseract import temporal, mock, metrics, evaluation
from tesseract.rebalancing import PositiveRateRebalancer


class TestRebalancers(unittest.TestCase):
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

    def test_positive_rate_rebalancer(self):
        for clf in (self.svm, self.rf):
            pr_rebalancer = PositiveRateRebalancer(0.5)
            results = evaluation.fit_predict_update(
                clf, self.X_train, self.X_tests,
                self.y_train, self.y_tests,
                self.t_train, self.t_tests,
                rebalancers=[pr_rebalancer])

            metrics.print_metrics(results)
