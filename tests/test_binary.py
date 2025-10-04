import numpy as np
from pytest import approx
from clustering_metrics.metrics import ConfusionMatrix2


def test_0000():
    m = (0, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

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


def test_1000():
    m = (1, 0, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    approx(h, 1.0, 4)
    approx(c, 1.0, 4)
    approx(v, 1.0, 4)

    approx(cm.dice_coeff(), 1.0, 4)
    approx(cm.ochiai_coeff(), 1.0, 4)
    approx(cm.ochiai_coeff_adj(), 0.5, 4)
    approx(cm.matthews_corr(), 0.5, 4)
    approx(cm.mp_corr(), 0.5, 4)
    approx(cm.kappa(), 0.5, 4)

    approx(cm.loevinger_coeff(), 0.5, 4)
    approx(cm.cole_coeff(), 0.5, 4)
    approx(cm.yule_q(), 1.0, 4)
    approx(cm.yule_y(), 1.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_0100():
    m = (0, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    approx(h, 1.0, 4)
    approx(c, 1.0, 4)
    approx(v, 1.0, 4)

    approx(cm.dice_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff_adj(), 0.0, 4)
    approx(cm.matthews_corr(), -0.5, 4)
    approx(cm.mp_corr(), -0.5, 4)
    approx(cm.kappa(), 0.0, 4)

    approx(cm.loevinger_coeff(), 0.0, 4)
    approx(cm.cole_coeff(), -0.5, 4)
    approx(cm.yule_y(), -1.0, 4)
    approx(cm.yule_q(), -1.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_0010():
    m = (0, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    approx(h, 1.0, 4)
    approx(c, 1.0, 4)
    approx(v, 1.0, 4)

    assert np.isnan(cm.dice_coeff())
    assert np.isnan(cm.ochiai_coeff())
    approx(cm.ochiai_coeff_adj(), 0.5, 4)
    approx(cm.matthews_corr(), 0.5, 4)
    approx(cm.mp_corr(), 0.5, 4)
    approx(cm.kappa(), 0.5, 4)

    approx(cm.loevinger_coeff(), 0.5, 4)
    approx(cm.cole_coeff(), 0.5, 4)
    approx(cm.yule_q(), 1.0, 4)
    approx(cm.yule_y(), 1.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_0001():
    m = (0, 0, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    h, c, v = cm.entropy_scores()
    approx(h, 1.0, 4)
    approx(c, 1.0, 4)
    approx(v, 1.0, 4)

    approx(cm.dice_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff_adj(), 0.0, 4)
    approx(cm.matthews_corr(), -0.5, 4)
    approx(cm.mp_corr(), -0.5, 4)
    approx(cm.kappa(), 0.0, 4)

    approx(cm.loevinger_coeff(), 0.0, 4)
    approx(cm.cole_coeff(), -0.5, 4)
    approx(cm.yule_y(), -1.0, 4)
    approx(cm.yule_q(), -1.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_1010():
    m = (1, 0, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 2.0, 4)
    approx(cm.g_score(), 2.7726, 4)

    approx(cm.dice_coeff(), 1.0, 4)
    approx(cm.ochiai_coeff(), 1.0, 4)
    approx(cm.ochiai_coeff_adj(), 1.0, 4)
    approx(cm.matthews_corr(), 1.0, 4)
    approx(cm.mp_corr(), 1.0, 4)
    approx(cm.kappa(), 1.0, 4)

    approx(cm.loevinger_coeff(), 1.0, 4)
    approx(cm.cole_coeff(), 1.0, 4)
    approx(cm.yule_q(), 1.0, 4)
    approx(cm.yule_y(), 1.0, 4)
    approx(cm.informedness(), 1.0, 4)
    approx(cm.markedness(), 1.0, 4)


def test_1100():
    m = (1, 1, 0, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    approx(cm.dice_coeff(), 0.6667, 4)
    approx(cm.ochiai_coeff(), 0.7071, 4)
    approx(cm.ochiai_coeff_adj(), 0.0, 4)
    approx(cm.matthews_corr(), 0.0, 4)
    approx(cm.mp_corr(), 0.0, 4)
    approx(cm.kappa(), 0.0, 4)

    approx(cm.loevinger_coeff(), 0.0, 4)
    approx(cm.cole_coeff(), 0.0, 4)
    approx(cm.yule_q(), 0.0, 4)
    approx(cm.yule_y(), 0.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_0011():
    m = (0, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    approx(cm.dice_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff_adj(), 0.0, 4)
    approx(cm.matthews_corr(), 0.0, 4)
    approx(cm.mp_corr(), 0.0, 4)
    approx(cm.kappa(), 0.0, 4)

    approx(cm.loevinger_coeff(), 0.0, 4)
    approx(cm.cole_coeff(), 0.0, 4)
    approx(cm.yule_q(), 0.0, 4)
    approx(cm.yule_y(), 0.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_0101():
    m = (0, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 2.0, 4)
    approx(cm.g_score(), 2.7726, 4)

    approx(cm.dice_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff_adj(), -1.0, 4)
    approx(cm.matthews_corr(), -1.0, 4)
    approx(cm.mp_corr(), -1.0, 4)
    approx(cm.kappa(), -1.0, 4)

    approx(cm.loevinger_coeff(), -1.0, 4)
    approx(cm.cole_coeff(), -1.0, 4)
    approx(cm.yule_q(), -1.0, 4)
    approx(cm.yule_y(), -1.0, 4)
    approx(cm.informedness(), -1.0, 4)
    approx(cm.markedness(), -1.0, 4)


def test_1001():
    m = (1, 0, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    approx(cm.dice_coeff(), 0.6667, 4)
    approx(cm.ochiai_coeff(), 0.7071, 4)
    approx(cm.ochiai_coeff_adj(), 0.0, 4)
    approx(cm.matthews_corr(), 0.0, 4)
    approx(cm.mp_corr(), 0.0, 4)
    approx(cm.kappa(), 0.0, 4)

    approx(cm.loevinger_coeff(), 0.0, 4)
    approx(cm.cole_coeff(), 0.0, 4)
    approx(cm.yule_q(), 0.0, 4)
    approx(cm.yule_y(), 0.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_0110():
    m = (0, 1, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    approx(cm.dice_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff_adj(), 0.0, 4)
    approx(cm.matthews_corr(), 0.0, 4)
    approx(cm.mp_corr(), 0.0, 4)
    approx(cm.kappa(), 0.0, 4)

    approx(cm.loevinger_coeff(), 0.0, 4)
    approx(cm.cole_coeff(), 0.0, 4)
    approx(cm.yule_q(), 0.0, 4)
    approx(cm.yule_y(), 0.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)


def test_0111():
    m = (0, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.75, 4)
    approx(cm.g_score(), 1.0465, 4)

    approx(cm.dice_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff(), 0.0, 4)
    approx(cm.ochiai_coeff_adj(), -0.5, 4)
    approx(cm.matthews_corr(), -0.5, 4)
    approx(cm.mp_corr(), -0.5, 4)
    approx(cm.kappa(), -0.5, 4)

    approx(cm.loevinger_coeff(), -0.5, 4)
    approx(cm.cole_coeff(), -1.0, 4)
    approx(cm.yule_q(), -1.0, 4)
    approx(cm.yule_y(), -1.0, 4)
    approx(cm.informedness(), -0.5, 4)
    approx(cm.markedness(), -0.5, 4)


def test_1011():
    m = (1, 0, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.75, 4)
    approx(cm.g_score(), 1.0465, 4)

    approx(cm.dice_coeff(), 0.6667, 4)
    approx(cm.ochiai_coeff(), 0.7071, 4)
    approx(cm.ochiai_coeff_adj(), 0.4459, 4)
    approx(cm.matthews_corr(), 0.5, 4)
    approx(cm.mp_corr(), 0.5, 4)
    approx(cm.kappa(), 0.4, 4)

    approx(cm.loevinger_coeff(), 1.0, 4)
    approx(cm.cole_coeff(), 1.0, 4)
    approx(cm.yule_q(), 1.0, 4)
    approx(cm.yule_y(), 1.0, 4)
    approx(cm.informedness(), 0.5, 4)
    approx(cm.markedness(), 0.5, 4)


def test_1101():
    m = (1, 1, 0, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.75, 4)
    approx(cm.g_score(), 1.0465, 4)

    approx(cm.dice_coeff(), 0.5, 4)
    approx(cm.ochiai_coeff(), 0.5, 4)
    approx(cm.ochiai_coeff_adj(), -0.5, 4)
    approx(cm.matthews_corr(), -0.5, 4)
    approx(cm.mp_corr(), -0.5, 4)
    approx(cm.kappa(), -0.5, 4)

    approx(cm.loevinger_coeff(), -0.5, 4)
    approx(cm.cole_coeff(), -1.0, 4)
    approx(cm.yule_q(), -1.0, 4)
    approx(cm.yule_y(), -1.0, 4)
    approx(cm.informedness(), -0.5, 4)
    approx(cm.markedness(), -0.5, 4)


def test_1110():
    m = (1, 1, 1, 0)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.75, 4)
    approx(cm.g_score(), 1.0465, 4)

    approx(cm.dice_coeff(), 0.6667, 4)
    approx(cm.ochiai_coeff(), 0.7071, 4)
    approx(cm.ochiai_coeff_adj(), 0.4459, 4)
    approx(cm.matthews_corr(), 0.5, 4)
    approx(cm.mp_corr(), 0.5, 4)
    approx(cm.kappa(), 0.4, 4)

    approx(cm.loevinger_coeff(), 1.0, 4)
    approx(cm.cole_coeff(), 1.0, 4)
    approx(cm.yule_q(), 1.0, 4)
    approx(cm.yule_y(), 1.0, 4)
    approx(cm.informedness(), 0.5, 4)
    approx(cm.markedness(), 0.5, 4)


def test_1111():
    m = (1, 1, 1, 1)
    cm = ConfusionMatrix2.from_ccw(*m)
    approx(cm.chisq_score(), 0.0, 4)
    approx(cm.g_score(), 0.0, 4)

    approx(cm.dice_coeff(), 0.5, 4)
    approx(cm.ochiai_coeff(), 0.5, 4)
    approx(cm.ochiai_coeff_adj(), 0.0, 4)
    approx(cm.matthews_corr(), 0.0, 4)
    approx(cm.mp_corr(), 0.0, 4)
    approx(cm.kappa(), 0.0, 4)

    approx(cm.loevinger_coeff(), 0.0, 4)
    approx(cm.cole_coeff(), 0.0, 4)
    approx(cm.yule_q(), 0.0, 4)
    approx(cm.yule_y(), 0.0, 4)
    approx(cm.informedness(), 0.0, 4)
    approx(cm.markedness(), 0.0, 4)
