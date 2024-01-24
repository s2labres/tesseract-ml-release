# -*- coding: utf-8 -*-

"""
test_mock.py
~~~~~~~

Unit tests for testing mock.py.

"""
import unittest

from tesseract import mock


class TestMock(unittest.TestCase):
    def test_generate_binary_test_data(self):
        X, y, t = mock.generate_binary_test_data(10000, '2016')
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(y), len(t))

    def test_generate_time_data(self):
        expected = ['2012-09-22', '2012-11-26', '2012-09-11', '2012-07-15']
        dates = mock.generate_time_data(4, '2012', random_state=22)
        actual = [d.strftime('%Y-%m-%d') for d in dates]
        self.assertEqual(expected, actual)

        expected = ['2012-11-07', '2011-12-20', '2015-11-08', '2011-09-13']
        dates = mock.generate_time_data(4, '2010', '2016', random_state=22)
        actual = [d.strftime('%Y-%m-%d') for d in dates]
        self.assertEqual(expected, actual)

        years = (2010, 2011, 2012, 2013, 2014)
        dates = mock.generate_time_data(10000, '2010', '2014-12-31')
        for date in dates:
            self.assertIn(date.year, years)

        expected = {2010}, {1}, {1}
        dates = mock.generate_time_data(10000, '2010-01-01', '2010-01-02')
        actual = (set(d.year for d in dates),
                  set(d.month for d in dates),
                  set(d.day for d in dates))
        self.assertEqual(expected, actual)
