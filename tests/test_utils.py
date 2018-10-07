__author__ = 'escherba'

import unittest
import numpy as np
# from clustering_metrics.fent import minmaxr
from clustering_metrics.utils import sort_by_length


class TestUtils(unittest.TestCase):

    # def test_minmaxr_1(self):
    #     arr = [2, 3, 4]
    #     amin, amax = minmaxr(arr)
    #     self.assertAlmostEqual(2.0, amin)
    #     self.assertAlmostEqual(4.0, amax)

    # def test_minmaxr_2(self):
    #     arr = []
    #     amin, amax = minmaxr(arr)
    #     self.assertTrue(np.isinf(amin))
    #     self.assertTrue(np.isinf(amin))
    #     self.assertGreater(amin, amax)

    def test_sort_by_length(self):
        """Should be able to sort a list of lists by length of sublists"""
        test_case = ["abraca", "abracadabra", "a", "aba"]
        result = sort_by_length(test_case)
        self.assertEqual(list(result), ["abracadabra", "abraca", "aba", "a"])


if __name__ == '__main__':
    unittest.main()
