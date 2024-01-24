# -*- coding: utf-8 -*-

"""
test_viz.py
~~~~~~~~~~~

Unit tests for testing viz.py.

"""
import unittest

from tesseract import mock, metrics


class TestViz(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.t = mock.generate_binary_test_data(10000, '2012')

    # def test_plot_by_time(self):
    #     viz.plot_by_time(self.y, self.t, 'day', 'line')
    #     viz.plot_by_time(self.y, self.t, 'week', 'line')
    #     viz.plot_by_time(self.y, self.t, 'month', 'line')
    #     viz.plot_by_time(self.y, self.t, 'month', 'bar')
    #     viz.plot_by_time(self.y, self.t, 'quarter', 'bar')

    def test_summarize(self):
        metrics.summarize(self.y)
