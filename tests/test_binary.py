import numpy as np
from nose.tools import assert_almost_equal, assert_true
from clustering_metrics.metrics import ConfusionMatrix2


def test_0000():
    m = (0, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    assert_true(np.isnan(cm.dice_coeff()))
    assert_true(np.isnan(cm.ochiai_coeff()))
    assert_true(np.isnan(cm.ochiai_coeff_adj()))
    assert_true(np.isnan(cm.matthews_corr()))
    assert_true(np.isnan(cm.mp_corr()))
    assert_true(np.isnan(cm.kappa()))

    assert_true(np.isnan(cm.loevinger_coeff()))
    assert_true(np.isnan(cm.cole_coeff()))
    assert_true(np.isnan(cm.yule_q()))
    assert_true(np.isnan(cm.yule_y()))
    assert_true(np.isnan(cm.informedness()))
    assert_true(np.isnan(cm.markedness()))


def test_1000():
    m = (1, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_almost_equal(cm.dice_coeff(), 1.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 1.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.5, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.mp_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.5, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.5, 4)
    assert_almost_equal(cm.cole_coeff(), 0.5, 4)
    assert_almost_equal(cm.yule_q(), 1.0, 4)
    assert_almost_equal(cm.yule_y(), 1.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_0100():
    m = (0, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_almost_equal(cm.dice_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.mp_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.0, 4)
    assert_almost_equal(cm.cole_coeff(), -0.5, 4)
    assert_almost_equal(cm.yule_y(), -1.0, 4)
    assert_almost_equal(cm.yule_q(), -1.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_0010():
    m = (0, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_true(np.isnan(cm.dice_coeff()))
    assert_true(np.isnan(cm.ochiai_coeff()))
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.5, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.mp_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.5, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.5, 4)
    assert_almost_equal(cm.cole_coeff(), 0.5, 4)
    assert_almost_equal(cm.yule_q(), 1.0, 4)
    assert_almost_equal(cm.yule_y(), 1.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_0001():
    m = (0, 0, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    assert_almost_equal(h, 1.0, 4)
    assert_almost_equal(c, 1.0, 4)
    assert_almost_equal(v, 1.0, 4)

    assert_almost_equal(cm.dice_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.mp_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.0, 4)
    assert_almost_equal(cm.cole_coeff(), -0.5, 4)
    assert_almost_equal(cm.yule_y(), -1.0, 4)
    assert_almost_equal(cm.yule_q(), -1.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_1010():
    m = (1, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 2.0, 4)
    assert_almost_equal(cm.g_score(), 2.7726, 4)

    assert_almost_equal(cm.dice_coeff(), 1.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 1.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 1.0, 4)
    assert_almost_equal(cm.matthews_corr(), 1.0, 4)
    assert_almost_equal(cm.mp_corr(), 1.0, 4)
    assert_almost_equal(cm.kappa(), 1.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 1.0, 4)
    assert_almost_equal(cm.cole_coeff(), 1.0, 4)
    assert_almost_equal(cm.yule_q(), 1.0, 4)
    assert_almost_equal(cm.yule_y(), 1.0, 4)
    assert_almost_equal(cm.informedness(), 1.0, 4)
    assert_almost_equal(cm.markedness(), 1.0, 4)


def test_1100():
    m = (1, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    assert_almost_equal(cm.dice_coeff(), 0.6667, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.7071, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.0, 4)
    assert_almost_equal(cm.cole_coeff(), 0.0, 4)
    assert_almost_equal(cm.yule_q(), 0.0, 4)
    assert_almost_equal(cm.yule_y(), 0.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_0011():
    m = (0, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    assert_almost_equal(cm.dice_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.0, 4)
    assert_almost_equal(cm.cole_coeff(), 0.0, 4)
    assert_almost_equal(cm.yule_q(), 0.0, 4)
    assert_almost_equal(cm.yule_y(), 0.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_0101():
    m = (0, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 2.0, 4)
    assert_almost_equal(cm.g_score(), 2.7726, 4)

    assert_almost_equal(cm.dice_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), -1.0, 4)
    assert_almost_equal(cm.matthews_corr(), -1.0, 4)
    assert_almost_equal(cm.mp_corr(), -1.0, 4)
    assert_almost_equal(cm.kappa(), -1.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), -1.0, 4)
    assert_almost_equal(cm.cole_coeff(), -1.0, 4)
    assert_almost_equal(cm.yule_q(), -1.0, 4)
    assert_almost_equal(cm.yule_y(), -1.0, 4)
    assert_almost_equal(cm.informedness(), -1.0, 4)
    assert_almost_equal(cm.markedness(), -1.0, 4)


def test_1001():
    m = (1, 0, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    assert_almost_equal(cm.dice_coeff(), 0.6667, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.7071, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.0, 4)
    assert_almost_equal(cm.cole_coeff(), 0.0, 4)
    assert_almost_equal(cm.yule_q(), 0.0, 4)
    assert_almost_equal(cm.yule_y(), 0.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_0110():
    m = (0, 1, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    assert_almost_equal(cm.dice_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.0, 4)
    assert_almost_equal(cm.cole_coeff(), 0.0, 4)
    assert_almost_equal(cm.yule_q(), 0.0, 4)
    assert_almost_equal(cm.yule_y(), 0.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)


def test_0111():
    m = (0, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)

    assert_almost_equal(cm.dice_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.0, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), -0.5, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.mp_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), -0.5, 4)

    assert_almost_equal(cm.loevinger_coeff(), -0.5, 4)
    assert_almost_equal(cm.cole_coeff(), -1.0, 4)
    assert_almost_equal(cm.yule_q(), -1.0, 4)
    assert_almost_equal(cm.yule_y(), -1.0, 4)
    assert_almost_equal(cm.informedness(), -0.5, 4)
    assert_almost_equal(cm.markedness(), -0.5, 4)


def test_1011():
    m = (1, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)

    assert_almost_equal(cm.dice_coeff(), 0.6667, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.7071, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.4459, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.mp_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.4, 4)

    assert_almost_equal(cm.loevinger_coeff(), 1.0, 4)
    assert_almost_equal(cm.cole_coeff(), 1.0, 4)
    assert_almost_equal(cm.yule_q(), 1.0, 4)
    assert_almost_equal(cm.yule_y(), 1.0, 4)
    assert_almost_equal(cm.informedness(), 0.5, 4)
    assert_almost_equal(cm.markedness(), 0.5, 4)


def test_1101():
    m = (1, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)

    assert_almost_equal(cm.dice_coeff(), 0.5, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.5, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), -0.5, 4)
    assert_almost_equal(cm.matthews_corr(), -0.5, 4)
    assert_almost_equal(cm.mp_corr(), -0.5, 4)
    assert_almost_equal(cm.kappa(), -0.5, 4)

    assert_almost_equal(cm.loevinger_coeff(), -0.5, 4)
    assert_almost_equal(cm.cole_coeff(), -1.0, 4)
    assert_almost_equal(cm.yule_q(), -1.0, 4)
    assert_almost_equal(cm.yule_y(), -1.0, 4)
    assert_almost_equal(cm.informedness(), -0.5, 4)
    assert_almost_equal(cm.markedness(), -0.5, 4)


def test_1110():
    m = (1, 1, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.75, 4)
    assert_almost_equal(cm.g_score(), 1.0465, 4)

    assert_almost_equal(cm.dice_coeff(), 0.6667, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.7071, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.4459, 4)
    assert_almost_equal(cm.matthews_corr(), 0.5, 4)
    assert_almost_equal(cm.mp_corr(), 0.5, 4)
    assert_almost_equal(cm.kappa(), 0.4, 4)

    assert_almost_equal(cm.loevinger_coeff(), 1.0, 4)
    assert_almost_equal(cm.cole_coeff(), 1.0, 4)
    assert_almost_equal(cm.yule_q(), 1.0, 4)
    assert_almost_equal(cm.yule_y(), 1.0, 4)
    assert_almost_equal(cm.informedness(), 0.5, 4)
    assert_almost_equal(cm.markedness(), 0.5, 4)


def test_1111():
    m = (1, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    assert_almost_equal(cm.chisq_score(), 0.0, 4)
    assert_almost_equal(cm.g_score(), 0.0, 4)

    assert_almost_equal(cm.dice_coeff(), 0.5, 4)
    assert_almost_equal(cm.ochiai_coeff(), 0.5, 4)
    assert_almost_equal(cm.ochiai_coeff_adj(), 0.0, 4)
    assert_almost_equal(cm.matthews_corr(), 0.0, 4)
    assert_almost_equal(cm.mp_corr(), 0.0, 4)
    assert_almost_equal(cm.kappa(), 0.0, 4)

    assert_almost_equal(cm.loevinger_coeff(), 0.0, 4)
    assert_almost_equal(cm.cole_coeff(), 0.0, 4)
    assert_almost_equal(cm.yule_q(), 0.0, 4)
    assert_almost_equal(cm.yule_y(), 0.0, 4)
    assert_almost_equal(cm.informedness(), 0.0, 4)
    assert_almost_equal(cm.markedness(), 0.0, 4)
