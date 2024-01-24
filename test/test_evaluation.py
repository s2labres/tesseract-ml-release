# -*- coding: utf-8 -*-

"""
test_evaluation.py
~~~~~~~~~~~~~~~~~~

Unit tests used to help redesign the typical Tesseract workflow.
"""

import unittest

from sklearn.svm import SVC

from tesseract import temporal, mock, metrics, evaluation


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        # Test partitions of 1 year
        X, y, t = mock.generate_binary_test_data(10000, '2020')

        splits = temporal.time_aware_train_test_split(
            X, y, t, train_size=6, test_size=2,
            granularity='month', start_date='2020')
        X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

        self.X_train = X_train
        self.y_train = y_train
        self.X_tests = X_tests
        self.y_tests = y_tests
        self.t_train = t_train
        self.t_tests = t_tests

        self.clf = SVC(kernel='linear', probability=True)
        self.clf.fit(X_train, y_train)

    def test_use_case_1(self):
        # Predict each test period yourself and get individual results
        for i, (X_test, y_true) in enumerate(zip(self.X_tests, self.y_tests)):
            y_pred = self.clf.predict(X_test)

            print('Test period {}'.format(i))
            results = metrics.calculate_metrics(y_true, y_pred)
            metrics.print_metrics(results, header=False)

    def test_use_case_2(self):
        # Keep a running data structure for results
        results = {}
        for i, (X_test, y_true) in enumerate(zip(self.X_tests, self.y_tests)):
            y_pred = self.clf.predict(X_test)

            results = metrics.calculate_metrics(
                y_true, y_pred, existing=results)
        metrics.print_metrics(results)

    def test_use_case_3(self):
        # Use a library method to run the entire prediction
        y_preds = evaluation.predict(self.clf, self.X_tests)
        results = metrics.calculate_metrics(self.y_tests, y_preds, periods=3)
        metrics.print_metrics(results)

    def test_use_case_4(self):
        # Parallelising computation of test periods
        y_preds = evaluation.predict(self.clf, self.X_tests, nproc=3)
        results = metrics.calculate_metrics(self.y_tests, y_preds, periods=-1)
        metrics.print_metrics(results)

    def test_use_case_5(self):
        # Forcing output to be labels rather than probabilities
        y_preds = evaluation.predict(
            self.clf, self.X_tests, labels_only=True)
        results = metrics.calculate_metrics(self.y_tests, y_preds, periods=-1)
        metrics.print_metrics(results)
        print(metrics.aut(results, 'f1'))
        print(metrics.aut(results['f1']))

    def test_use_case_6(self):
        # Use full fit_predict_update to measure the performance
        results = evaluation.fit_predict_update(
            self.clf, self.X_train, self.X_tests,
            self.y_train, self.y_tests,
            self.t_train, self.t_tests)

        metrics.print_metrics(results)
        print(metrics.aut(results, 'f1'))
        print(metrics.aut(results['f1']))
