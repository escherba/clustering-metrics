"""
Test numerical accuracy of some implementations
"""

import unittest
import numpy as np
from clustering_metrics.metrics import ClusteringMetrics
# from clustering_metrics.fent import emi_from_margins as emi_fortran
from clustering_metrics.entropy import emi_from_margins as emi_cython


class TestAccuracy(unittest.TestCase):

    def test_emi_matlab(self):
        """Compare EMI values with reference MATLAB code

        http://www.mathworks.com/matlabcentral/fileexchange/33144-the-adjusted-mutual-information
        """

        ltrue = "11 11 11 11 11 11 11 10 10 10 10 13 13 13 13 13 13 13 13 13 12 \
        12 12 12 12 15 15 15 15 15 15 15 14 14 14 14 14 17 17 17 17 16 16 16 16 \
        16 16 19 19 19 19 19 19 19 18 18 18 18 18 18 18 20 20 20 20 20 20 1 1 1 \
        1 3 3 2 2 2 5 5 5 4 4 4 4 7 7 7 7 7 7 7 7 7 6 6 6 9 9 9 8 8".split()

        lpred = "1 19 19 13 2 20 20 8 12 5 17 10 10 13 15 20 20 6 9 8 9 10 15 \
        14 8 11 11 10 13 17 19 5 9 1 2 20 15 19 19 12 14 1 18 18 3 2 5 8 8 7 17 \
        17 17 16 11 11 14 17 16 6 8 13 17 1 3 7 9 9 1 5 18 13 17 13 12 20 11 4 \
        14 19 15 13 5 13 12 16 4 4 7 6 6 8 2 16 16 18 3 7 1 10".split()

        cm = ClusteringMetrics.from_labels(ltrue, lpred)
        ami = cm.adjusted_mutual_info()

        self.assertAlmostEqual(0.0352424389209073, ami, 12)

        # rmarg = np.asarray(cm.row_totals.values(), dtype=np.int64)
        # cmarg = np.asarray(cm.col_totals.values(), dtype=np.int64)

        # emi1 = emi_fortran(rmarg, cmarg)
        # emi2 = emi_cython(rmarg, cmarg)

        # self.assertAlmostEqual(emi1, emi2, 10)
