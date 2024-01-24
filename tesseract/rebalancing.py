# -*- coding: utf-8 -*-

"""
rebalancing.py
~~~~~~~~~~~~~~

# TODO | Add module description

"""

import numpy as np

from tesseract import spatial
from tesseract.evaluation import Stage


class Rebalancer(Stage):
    def alter_wrapper(self, clf, X_train, y_train, t_train, X_test,
                      y_test, t_test):
        # Pass parameters straight through to rebalance implementation
        rebalanced = self.alter(clf, X_train, y_train, t_train,
                                X_test, y_test, t_test)

        return np.array(rebalanced)

    def alter(self, clf, X_train, y_train, t_train, X_test, y_test, t_test):
        raise NotImplementedError('Rebalancer must be subclassed')


class PositiveRateRebalancer(Rebalancer):
    def __init__(self, min_pos_rate, max_pos_rate=None, noise_deviation=0.0,
                 fixed_size=False, schedule=1):
        super().__init__(schedule=schedule)
        self.min_pos_rate = min_pos_rate
        self.max_pos_rate = max_pos_rate
        self.noise_deviation = noise_deviation
        self.fixed_size = fixed_size

    def alter(self, clf, X_train, y_train, t_train, X_test, y_test, t_test):
        return spatial.downsample_set(
            X_train, y_train, t_train, self.min_pos_rate,
            self.max_pos_rate, self.noise_deviation, self.fixed_size)
