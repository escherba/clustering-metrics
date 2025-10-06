import unittest
import numpy as np
from clustering_metrics.metrics import ConfusionMatrix2


class BinaryTests(unittest.TestCase):

    def test_0000(self):
        m = (0, 0, 0, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        assert np.isnan(cm.dice_coeff())
        assert np.isnan(cm.ochiai_coeff())
        assert np.isnan(cm.ochiai_coeff_adj())
        assert np.isnan(cm.matthews_corr())
        assert np.isnan(cm.mp_corr())
        assert np.isnan(cm.kappa())

        assert np.isnan(cm.loevinger_coeff())
        assert np.isnan(cm.cole_coeff())
        assert np.isnan(cm.yule_q())
        assert np.isnan(cm.yule_y())
        assert np.isnan(cm.informedness())
        assert np.isnan(cm.markedness())


    def test_1000(self):
        m = (1, 0, 0, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        h, c, v = cm.entropy_scores()
        self.assertAlmostEqual(h, 1.0, 4)
        self.assertAlmostEqual(c, 1.0, 4)
        self.assertAlmostEqual(v, 1.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.5, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.kappa(), 0.5, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.yule_q(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 1.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_0100(self):
        m = (0, 1, 0, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        h, c, v = cm.entropy_scores()
        self.assertAlmostEqual(h, 1.0, 4)
        self.assertAlmostEqual(c, 1.0, 4)
        self.assertAlmostEqual(v, 1.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.kappa(), 0.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), -0.5, 4)
        self.assertAlmostEqual(cm.yule_y(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), -1.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_0010(self):
        m = (0, 0, 1, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        h, c, v = cm.entropy_scores()
        self.assertAlmostEqual(h, 1.0, 4)
        self.assertAlmostEqual(c, 1.0, 4)
        self.assertAlmostEqual(v, 1.0, 4)

        assert np.isnan(cm.dice_coeff())
        assert np.isnan(cm.ochiai_coeff())
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.5, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.kappa(), 0.5, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.yule_q(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 1.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_0001(self):
        m = (0, 0, 0, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        h, c, v = cm.entropy_scores()
        self.assertAlmostEqual(h, 1.0, 4)
        self.assertAlmostEqual(c, 1.0, 4)
        self.assertAlmostEqual(v, 1.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.kappa(), 0.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), -0.5, 4)
        self.assertAlmostEqual(cm.yule_y(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), -1.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_1010(self):
        m = (1, 0, 1, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 2.0, 4)
        self.assertAlmostEqual(cm.g_score(), 2.7726, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 1.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 1.0, 4)
        self.assertAlmostEqual(cm.mp_corr(), 1.0, 4)
        self.assertAlmostEqual(cm.kappa(), 1.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 1.0, 4)
        self.assertAlmostEqual(cm.informedness(), 1.0, 4)
        self.assertAlmostEqual(cm.markedness(), 1.0, 4)


    def test_1100(self):
        m = (1, 1, 0, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.6667, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.7071, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.kappa(), 0.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 0.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_0011(self):
        m = (0, 0, 1, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.kappa(), 0.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 0.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_0101(self):
        m = (0, 1, 0, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 2.0, 4)
        self.assertAlmostEqual(cm.g_score(), 2.7726, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), -1.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), -1.0, 4)
        self.assertAlmostEqual(cm.mp_corr(), -1.0, 4)
        self.assertAlmostEqual(cm.kappa(), -1.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), -1.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), -1.0, 4)
        self.assertAlmostEqual(cm.informedness(), -1.0, 4)
        self.assertAlmostEqual(cm.markedness(), -1.0, 4)


    def test_1001(self):
        m = (1, 0, 0, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.6667, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.7071, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.kappa(), 0.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 0.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_0110(self):
        m = (0, 1, 1, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.kappa(), 0.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 0.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)


    def test_0111(self):
        m = (0, 1, 1, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.75, 4)
        self.assertAlmostEqual(cm.g_score(), 1.0465, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), -0.5, 4)
        self.assertAlmostEqual(cm.matthews_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.kappa(), -0.5, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), -0.5, 4)
        self.assertAlmostEqual(cm.cole_coeff(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), -1.0, 4)
        self.assertAlmostEqual(cm.informedness(), -0.5, 4)
        self.assertAlmostEqual(cm.markedness(), -0.5, 4)


    def test_1011(self):
        m = (1, 0, 1, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.75, 4)
        self.assertAlmostEqual(cm.g_score(), 1.0465, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.6667, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.7071, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.4459, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.kappa(), 0.4, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 1.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.5, 4)
        self.assertAlmostEqual(cm.markedness(), 0.5, 4)


    def test_1101(self):
        m = (1, 1, 0, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.75, 4)
        self.assertAlmostEqual(cm.g_score(), 1.0465, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), -0.5, 4)
        self.assertAlmostEqual(cm.matthews_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), -0.5, 4)
        self.assertAlmostEqual(cm.kappa(), -0.5, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), -0.5, 4)
        self.assertAlmostEqual(cm.cole_coeff(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), -1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), -1.0, 4)
        self.assertAlmostEqual(cm.informedness(), -0.5, 4)
        self.assertAlmostEqual(cm.markedness(), -0.5, 4)


    def test_1110(self):
        m = (1, 1, 1, 0)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.75, 4)
        self.assertAlmostEqual(cm.g_score(), 1.0465, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.6667, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.7071, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.4459, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.5, 4)
        self.assertAlmostEqual(cm.kappa(), 0.4, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 1.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 1.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.5, 4)
        self.assertAlmostEqual(cm.markedness(), 0.5, 4)


    def test_1111(self):
        m = (1, 1, 1, 1)
        cm = ConfusionMatrix2.from_ccw(*m)
        self.assertAlmostEqual(cm.chisq_score(), 0.0, 4)
        self.assertAlmostEqual(cm.g_score(), 0.0, 4)

        self.assertAlmostEqual(cm.dice_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.ochiai_coeff(), 0.5, 4)
        self.assertAlmostEqual(cm.ochiai_coeff_adj(), 0.0, 4)
        self.assertAlmostEqual(cm.matthews_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.mp_corr(), 0.0, 4)
        self.assertAlmostEqual(cm.kappa(), 0.0, 4)

        self.assertAlmostEqual(cm.loevinger_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.cole_coeff(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_q(), 0.0, 4)
        self.assertAlmostEqual(cm.yule_y(), 0.0, 4)
        self.assertAlmostEqual(cm.informedness(), 0.0, 4)
        self.assertAlmostEqual(cm.markedness(), 0.0, 4)
