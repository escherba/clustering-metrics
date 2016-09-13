"""
Randomized tests for invariant properties of some clustering metrics
"""

import numpy as np
import warnings
from clustering_metrics.metrics import ClusteringMetrics,  ConfusionMatrix2, \
    harmonic_mean, _div, adjusted_rand_score, mutual_info_score, \
    adjusted_mutual_info_score, geometric_mean
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_almost_equal, assert_true, assert_equal
from sklearn.metrics import \
    homogeneity_completeness_v_measure as sklearn_hcv, \
    adjusted_rand_score as sklearn_ari, \
    mutual_info_score as sklearn_mi, \
    adjusted_mutual_info_score as sklearn_ami
from clustering_metrics.hungarian import linear_sum_assignment


def assignment_score_slow(cm, normalize=True, rpad=False, cpad=False):
    """Calls Python/Numpy implementation of the Hungarian method

    Testing version (uses SciPy's implementation)

    """
    cost_matrix = -cm.to_array(rpad=rpad, cpad=cpad)
    ris, cis = linear_sum_assignment(cost_matrix)
    score = -cost_matrix[ris, cis].sum()
    if normalize:
        score = _div(score, cm.grand_total)
    return score


def check_with_nans(num1, num2, places=None, msg=None, delta=None, ensure_nans=True):
    nancheck_msg = "NaN check failed for '%s'" % msg
    if np.isnan(num1):
        if ensure_nans:
            assert_true(np.isnan(num2), msg=nancheck_msg)
        elif not np.isnan(num2):
            warnings.warn(nancheck_msg)
    elif np.isnan(num2):
        if ensure_nans:
            assert_true(np.isnan(num1), msg=nancheck_msg)
        elif not np.isnan(num1):
            warnings.warn(nancheck_msg)
    else:
        assert_almost_equal(num1, num2, places=places, msg=msg, delta=delta)


def test_m1():
    """M1 model
    """
    t2 = ClusteringMetrics(rows=10 * np.ones((2, 2), dtype=int))
    t8 = ClusteringMetrics(rows=10 * np.ones((8, 8), dtype=int))

    assert_almost_equal(0.0, t2.vi_similarity_m1())
    assert_almost_equal(0.0, t8.vi_similarity_m1())

    assert_almost_equal(0.0, t2.split_join_similarity_m1())
    assert_almost_equal(0.0, t8.split_join_similarity_m1())

    assert_almost_equal(0.0, t2.assignment_score_m1())
    assert_almost_equal(0.0, t8.assignment_score_m1())


def test_RxC_general():
    """General conteingency-table mathods
    """
    for _ in xrange(100):
        size = np.random.randint(4, 100)
        a = np.random.randint(low=0, high=np.random.randint(low=2, high=100),
                              size=(size,))
        b = np.random.randint(low=0, high=np.random.randint(low=2, high=100),
                              size=(size,))
        cm = ClusteringMetrics.from_labels(a, b)

        assert_almost_equal(
            cm.assignment_score(model=None),
            assignment_score_slow(cm, rpad=False, cpad=False))

        assert_almost_equal(
            cm.assignment_score(model=None),
            assignment_score_slow(cm, rpad=True, cpad=True))

        for model in ['m1', 'm2r', 'm2c', 'm3']:

            assert_almost_equal(
                cm.grand_total,
                sum(cm.expected(model=model).itervalues()))

            assert_almost_equal(
                cm.assignment_score(model=model),
                cm.adjust_to_null(cm.assignment_score, model=model)[0])

            assert_almost_equal(
                cm.split_join_similarity(model=model),
                cm.adjust_to_null(cm.split_join_similarity, model=model)[0])


def test_RxC_metrics():
    """Alternative implementations should coincide for RxC matrices
    """
    for _ in xrange(100):
        ltrue = np.random.randint(low=0, high=5, size=(20,))
        lpred = np.random.randint(low=0, high=5, size=(20,))
        cm = ClusteringMetrics.from_labels(ltrue, lpred)

        # homogeneity, completeness, V-measure
        expected_v = cm.vi_similarity_m3()
        expected_hcv = sklearn_hcv(ltrue, lpred)
        actual_hcv = cm.entropy_scores()
        assert_array_almost_equal(actual_hcv, expected_hcv)
        assert_array_almost_equal(actual_hcv[2], expected_v)

        # mutual information score
        expected_mi = sklearn_mi(ltrue, lpred)
        actual_mi = mutual_info_score(ltrue, lpred)
        assert_array_almost_equal(actual_mi, expected_mi)

        # adjusted mutual information
        expected_ami = sklearn_ami(ltrue, lpred)
        actual_ami = adjusted_mutual_info_score(ltrue, lpred)
        assert_array_almost_equal(actual_ami, expected_ami)

        # adjusted rand index
        expected_ari = sklearn_ari(ltrue, lpred)
        actual_ari = adjusted_rand_score(ltrue, lpred)
        assert_array_almost_equal(actual_ari, expected_ari)


def test_2x2_invariants():
    """Alternative implementations should coincide for 2x2 matrices
    """

    for _ in xrange(100):
        cm = ConfusionMatrix2.from_random_counts(low=0, high=10)

        # object idempotency
        assert_equal(
            cm.to_ccw(),
            ConfusionMatrix2.from_ccw(*cm.to_ccw()).to_ccw(),
            msg="must be able to convert to tuple and create from tuple")

        # pairwise H, C, V
        h, c, v = cm.pairwise_hcv()[:3]
        check_with_nans(v, geometric_mean(h, c), ensure_nans=False)

        # informedness
        actual_info = cm.informedness()
        expected_info_1 = cm.TPR() + cm.TNR() - 1.0
        expected_info_2 = cm.TPR() - cm.FPR()
        check_with_nans(actual_info, expected_info_1, 4, ensure_nans=False)
        check_with_nans(actual_info, expected_info_2, 4, ensure_nans=False)

        # markedness
        actual_mark = cm.markedness()
        expected_mark_1 = cm.PPV() + cm.NPV() - 1.0
        expected_mark_2 = cm.PPV() - cm.FOR()
        check_with_nans(actual_mark, expected_mark_1, 4, ensure_nans=False)
        check_with_nans(actual_mark, expected_mark_2, 4, ensure_nans=False)

        # matthews corr coeff
        # actual_mcc = cm.matthews_corr()
        # expected_mcc = geometric_mean(actual_info, actual_mark)
        # check_with_nans(actual_mcc, expected_mcc, 4, ensure_nans=False)

        # kappas
        actual_kappa = cm.kappa()

        # kappa is the same as harmonic mean of kappa components
        expected_kappa_1 = harmonic_mean(*cm.kappas()[:2])
        check_with_nans(actual_kappa, expected_kappa_1, 4, ensure_nans=False)

        # kappa is the same as accuracy adjusted for chance
        expected_kappa_2 = harmonic_mean(*cm.adjust_to_null(cm.accuracy, model='m3'))
        check_with_nans(actual_kappa, expected_kappa_2, 4, ensure_nans=False)

        # kappa is the same as Dice coeff adjusted for chance
        expected_kappa_3 = harmonic_mean(*cm.adjust_to_null(cm.dice_coeff, model='m3'))
        check_with_nans(actual_kappa, expected_kappa_3, 4, ensure_nans=False)

        # odds ratio and Yule's Q
        actual_odds_ratio = cm.DOR()
        actual_yule_q = cm.yule_q()
        expected_yule_q = _div(actual_odds_ratio - 1.0, actual_odds_ratio + 1.0)
        expected_odds_ratio = _div(cm.PLL(), cm.NLL())
        check_with_nans(actual_odds_ratio, expected_odds_ratio, 4, ensure_nans=False)
        check_with_nans(actual_yule_q, expected_yule_q, 4, ensure_nans=False)

        # F-score and Dice
        expected_f = harmonic_mean(cm.precision(), cm.recall())
        actual_f = cm.fscore()
        check_with_nans(expected_f, actual_f, 6)
        check_with_nans(expected_f, cm.dice_coeff(), 6, ensure_nans=False)

        # association coefficients (1)
        dice = cm.dice_coeff()
        expected_jaccard = _div(dice, 2.0 - dice)
        actual_jaccard = cm.jaccard_coeff()
        check_with_nans(actual_jaccard, expected_jaccard, 6, ensure_nans=False)

        # association coefficients (2)
        jaccard = cm.jaccard_coeff()
        expected_ss2 = _div(jaccard, 2.0 - jaccard)
        actual_ss2 = cm.sokal_sneath_coeff()
        check_with_nans(actual_ss2, expected_ss2, 6, ensure_nans=False)

        # adjusted ochiai
        actual = cm.ochiai_coeff_adj()
        expected = harmonic_mean(*cm.adjust_to_null(cm.ochiai_coeff, model='m3'))
        check_with_nans(actual, expected, 6, ensure_nans=False)
