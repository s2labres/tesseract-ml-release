# -*- coding: utf-8 -*-

"""
test_temporal.py
~~~~~~~~~~~~~~~~

Unit tests for temporal.py.

"""
import random
import unittest
from datetime import datetime

import numpy as np
from sklearn.svm import LinearSVC

from tesseract import temporal, mock, selection, evaluation


class TestTemporal(unittest.TestCase):
    def test_train(self):
        # Test partitions of 1 year
        X, y, t = mock.generate_binary_test_data(10000, '2020')
        splits = temporal.time_aware_train_test_split(
            X, y, t, 6, 2, granularity='month', start_date='2020')
        X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

        results = evaluation.fit_predict_update(LinearSVC(), X_train, X_tests,
                                                y_train, y_tests, t_train,
                                                t_tests)
        print(results)
        results = evaluation.fit_predict_update(
            LinearSVC(), X_train, X_tests, y_train, y_tests, t_train, t_tests)
        print(results)

    def test_time_aware_indexes(self):
        # Test partitions of 1 year
        t = np.array([datetime(2020, x, 1) for x in range(1, 13)])
        random.shuffle(t)
        train, tests = temporal.time_aware_indexes(
            t, 6, 2, granularity='month', start_date='2020')

        # Smoke tests
        self.assertEqual(6, len(train))
        self.assertEqual(3, len(tests))

        for test in tests:
            self.assertEqual(2, len(test))

        # Check partition is complete and non-destructive
        recreated = train + [x for sub in tests for x in sub]
        self.assertEqual(len(recreated), len(t))
        self.assertEqual(set(recreated), set(range(len(t))))

        t_train = t[train]
        t_tests = [t[index_set] for index_set in tests]

        # Check partition is history-aware
        for m in t_train:
            for n in t_tests[0]:
                self.assertTrue(m < n)

        for i in range(0, len(t_tests) - 1):
            for m in t_tests[i]:
                for n in t_tests[i + 1]:
                    self.assertTrue(m < n)

    def test_time_aware_train_test_split(self):
        # Test partitions of 1 year
        X, y, t = mock.generate_binary_test_data(10000, '2020')
        X_train, X_tests, y_train, y_tests, t_train, t_tests = \
            temporal.time_aware_train_test_split(
                X, y, t, 6, 2, granularity='month', start_date='2020')

        # Smoke tests
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_tests), len(y_tests))
        self.assertEqual(len(X_tests[0]), len(y_tests[0]))

        for i in range(len(X_tests)):
            self.assertEqual(len(X_tests[i]), len(y_tests[i]))

    def test_closest_to_hyperplane(self):
        narray = np.array([3, -1, 7, 2, 5, -4])
        indexes = selection.closest_to_hyperplane(narray, 2)
        self.assertTrue(all([1, 3] == indexes))
        self.assertTrue(all([-1, 2] == narray[indexes]))
