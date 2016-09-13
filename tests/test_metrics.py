import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
from math import sqrt
from itertools import izip
from nose.tools import assert_almost_equal, assert_true, assert_equal, assert_greater
from clustering_metrics.metrics import adjusted_rand_score, \
    homogeneity_completeness_v_measure, fentropy, \
    jaccard_similarity, ClusteringMetrics, \
    ConfusionMatrix2, geometric_mean, harmonic_mean, _div, cohen_kappa, \
    product_moment, mutual_info_score, \
    adjusted_mutual_info_score, emi_from_margins as emi_cython
from clustering_metrics.fent import emi_from_margins as emi_fortran


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


def _kappa(a, c, d, b):
    """An alternative implementation of Cohen's kappa (for testing)
    """
    n = a + b + c + d
    p1 = a + b
    p2 = a + c
    q1 = c + d
    q2 = b + d
    if a == n or b == n or c == n or d == n:
        # only one cell is non-zero
        return np.nan
    elif p1 == 0 or p2 == 0 or q1 == 0 or q2 == 0:
        # one row or column is zero, another non-zero
        return 0.0
    else:
        # no more than one cell is zero
        po = a + d
        pe = (p2 * p1 + q2 * q1) / float(n)
        return _div(po - pe, n - pe)


def _entropy_scores(cm):
    """Given a ClusteringMetrics object, calculate three entropy-based metrics

    (Alternative implementation for testing)
    """
    H_C = fentropy(cm.row_totals)
    H_K = fentropy(cm.col_totals)
    H_CK = sum(fentropy(col) for col in cm.iter_cols())
    H_KC = sum(fentropy(row) for row in cm.iter_rows())
    # The '<=' comparisons below both prevent division by zero errors
    # and ensure that the scores are non-negative.
    homogeneity = 0.0 if H_C <= H_CK else (H_C - H_CK) / H_C
    completeness = 0.0 if H_K <= H_KC else (H_K - H_KC) / H_K
    nmi_score = harmonic_mean(homogeneity, completeness)
    return homogeneity, completeness, nmi_score


def _talburt_wang_index(labels_true, labels_pred):
    """Alt. implementation of Talburt-Wang index for testing
    """
    V = set()
    A = set()
    B = set()
    for pair in izip(labels_true, labels_pred):
        V.add(pair)
        A.add(pair[0])
        B.add(pair[1])
    prod = len(A) * len(B)
    return np.nan if prod == 0 else sqrt(prod) / len(V)


def uniform_labelings_scores(score_func, n_samples, k_range, n_runs=10,
                             seed=42):
    # Compute score for random uniform cluster labelings
    random_labels = np.random.RandomState(seed).random_integers
    scores = np.zeros((len(k_range), n_runs))
    for i, k in enumerate(k_range):
        for j in range(n_runs):
            labels_a = random_labels(low=0, high=k - 1, size=n_samples)
            labels_b = random_labels(low=0, high=k - 1, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


def test_perfect():
    p1 = [['A', 'B', 'C']]
    p2 = [['A', 'B', 'C']]
    cm = ClusteringMetrics.from_partitions(p1, p2)
    assert_almost_equal(cm.assignment_score(), 1.0, 4)
    assert_almost_equal(cm.vi_similarity(), 1.0, 4)
    assert_almost_equal(cm.split_join_similarity(), 1.0, 4)
    assert_almost_equal(cm.talburt_wang_index(), 1.0, 4)
    assert_array_almost_equal(cm.entropy_scores(), (1.0,) * 3, 4)
    assert_array_almost_equal(cm.bc_metrics(), (1.0,) * 3, 4)
    assert_array_almost_equal(cm.muc_scores(), (1.0,) * 3, 4)


def test_diseq():
    """Linkage disequilibrium should equal precalculated value
    """
    cf = ConfusionMatrix2(rows=[[474, 142], [611, 773]])
    assert_almost_equal(cf.diseq_coeff(), 0.0699, 4)
    cf = ConfusionMatrix2(rows=[[331, 331], [6.6, 331]])
    assert_almost_equal(cf.cole_coeff(), 0.942, 3)


def test_adjusted_mutual_info_score():
    # Compute the Adjusted Mutual Information and test against known values
    labels_a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    labels_b = np.array([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2])

    # Mutual information
    mi_1 = mutual_info_score(labels_a, labels_b)
    assert_almost_equal(mi_1, 0.41022, 5)
    mi_2 = mutual_info_score(labels_b, labels_a)
    assert_almost_equal(mi_2, 0.41022, 5)

    # Expected mutual information
    cm = ClusteringMetrics.from_labels(labels_a, labels_b)
    row_totals = np.fromiter(cm.iter_row_totals(), dtype=np.int64)
    col_totals = np.fromiter(cm.iter_col_totals(), dtype=np.int64)
    emi_1a = emi_cython(row_totals, col_totals) / cm.grand_total
    emi_1b = emi_fortran(row_totals, col_totals) / cm.grand_total
    assert_almost_equal(emi_1a, 0.15042, 5)
    assert_almost_equal(emi_1b, 0.15042, 5)
    emi_2a = emi_cython(col_totals, row_totals) / cm.grand_total
    emi_2b = emi_fortran(col_totals, row_totals) / cm.grand_total
    assert_almost_equal(emi_2a, 0.15042, 5)
    assert_almost_equal(emi_2b, 0.15042, 5)

    # Adjusted mutual information (1)
    ami_1 = adjusted_mutual_info_score(labels_a, labels_b)
    assert_almost_equal(ami_1, 0.27502, 5)
    ami_2 = adjusted_mutual_info_score(labels_a, labels_b)
    assert_almost_equal(ami_2, 0.27502, 5)

    # Adjusted mutual information (2)
    ami_1 = adjusted_mutual_info_score([1, 1, 2, 2], [2, 2, 3, 3])
    assert_equal(ami_1, 1.0)
    ami_2 = adjusted_mutual_info_score([2, 2, 3, 3], [1, 1, 2, 2])
    assert_equal(ami_2, 1.0)

    # Test AMI with a very large array
    a110 = np.array([list(labels_a) * 110]).flatten()
    b110 = np.array([list(labels_b) * 110]).flatten()
    ami = adjusted_mutual_info_score(a110, b110)
    assert_almost_equal(ami, 0.37, 2)  # not accurate to more than 2 places


def test_jaccard_nan():
    """Returns NaN for empty set
    """
    sim = jaccard_similarity([], [])
    assert_true(np.isnan(sim))


def test_entropy_of_counts_zero():
    """Returns zero for empty set
    """
    val = fentropy([])
    assert_almost_equal(val, 0.0000, 4)


def test_perfectly_good_clustering():
    """Perfect separation
    """
    h, c, v = homogeneity_completeness_v_measure([0, 0, 1, 1], [1, 1, 0, 0])
    assert_almost_equal(h, 1.00, 2)
    assert_almost_equal(c, 1.00, 2)
    assert_almost_equal(v, 1.00, 2)


def test_perfectly_bad_clustering():
    """No separation
    """
    h, c, v = homogeneity_completeness_v_measure([0, 0, 1, 1], [1, 1, 1, 1])
    assert_almost_equal(h, 0.00, 2)
    assert_almost_equal(c, 1.00, 2)
    assert_almost_equal(v, 0.00, 2)


def test_homogeneous_but_not_complete_labeling():
    """homogeneous but not complete clustering
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 2, 2])
    assert_almost_equal(h, 1.00, 2)
    assert_almost_equal(c, 0.69, 2)
    assert_almost_equal(v, 0.81, 2)


def test_complete_but_not_homogeneous_labeling():
    """complete but not homogeneous clustering
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 1, 1])
    assert_almost_equal(h, 0.58, 2)
    assert_almost_equal(c, 1.00, 2)
    assert_almost_equal(v, 0.73, 2)


def test_not_complete_and_not_homogeneous_labeling():
    """neither complete nor homogeneous but not so bad either
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)


def test_non_consecutive_labels_std():
    """regression tests for labels with gaps
    """
    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 2, 2, 2],
        [0, 1, 0, 1, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)

    h, c, v = homogeneity_completeness_v_measure(
        [0, 0, 0, 1, 1, 1],
        [0, 4, 0, 4, 2, 2])
    assert_almost_equal(h, 0.67, 2)
    assert_almost_equal(c, 0.42, 2)
    assert_almost_equal(v, 0.52, 2)


def test_ari_nan():
    """Returns NaN for empty lists
    """
    ari = adjusted_rand_score([], [])
    assert_true(np.isnan(ari))


def test_non_consecutive_labels_ari():
    """regression tests for labels with gaps
    """
    ari_1 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2])
    ari_2 = adjusted_rand_score([0, 0, 0, 1, 1, 1], [0, 4, 0, 4, 2, 2])
    assert_almost_equal(ari_1, 0.24, 2)
    assert_almost_equal(ari_2, 0.24, 2)


def test_split_join():
    """test split-join and related metrics

    Example given in
    http://stats.stackexchange.com/a/25001/37267

    For two different clustering pairs below, one can be obtained from the other
    by moving only two points, {9, 10} for the first pair, and {11, 12} for the
    second pair. The split-join distance for the two pairs is thus the same.

    Mirkin and VI distance is larger for the first pair (C1 and C2). This is not
    a fault of these measures as the clusterings in C3 and C4 do appear to
    capture more information than in the case of C1 and C2, and so their
    similarities should be greater.
    """

    C1 = [{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}]
    C2 = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}]
    cm = ClusteringMetrics.from_partitions(C1, C2)
    assert_equal(cm.mirkin_mismatch_coeff(normalize=False), 56)
    assert_almost_equal(cm.vi_distance(normalize=False), 0.594, 3)
    assert_equal(cm.split_join_distance(normalize=False), 4)

    C3 = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}]
    C4 = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10, 11, 12}, {13, 14, 15, 16}]
    cm = ClusteringMetrics.from_partitions(C3, C4)
    assert_equal(cm.mirkin_mismatch_coeff(normalize=False), 40)
    assert_almost_equal(cm.vi_distance(normalize=False), 0.520, 3)
    assert_equal(cm.split_join_distance(normalize=False), 4)


def test_bc_metrics():
    """Examples 1 and 2, listing in Figure 9, Bagga & Baldwin (1998)
    """
    p1 = ["1 2 3 4 5".split(), "6 7".split(), "8 9 A B C".split()]

    p2 = ["1 2 3 4 5".split(), "6 7 8 9 A B C".split()]
    cm = ClusteringMetrics.from_partitions(p1, p2)
    assert_array_almost_equal(cm.bc_metrics()[:2], [0.76, 1.0], 2)
    assert_array_almost_equal(cm.muc_scores()[:2], [0.9, 1.0], 4)

    p2 = ["1 2 3 4 5 8 9 A B C".split(), "6 7".split()]
    cm = ClusteringMetrics.from_partitions(p1, p2)
    assert_array_almost_equal(cm.bc_metrics()[:2], [0.58, 1.0], 2)
    assert_array_almost_equal(cm.muc_scores()[:2], [0.9, 1.0], 4)


def test_mt_metrics():
    """Table 1 in Vilain et al. (1995)
    """

    # row 1
    p1 = ["A B C D".split()]
    p2 = ["A B".split(), "C D".split()]
    cm = ClusteringMetrics.from_partitions(p1, p2)
    assert_array_almost_equal(cm.muc_scores()[:2], [1.0, 0.6667], 4)

    # row 2
    p1 = ["A B".split(), "C D".split()]
    p2 = ["A B C D".split()]
    cm = ClusteringMetrics.from_partitions(p1, p2)
    assert_array_almost_equal(cm.muc_scores()[:2], [0.6667, 1.0], 4)

    # row 3
    p1 = ["A B C D".split()]
    p2 = ["A B C D".split()]
    cm = ClusteringMetrics.from_partitions(p1, p2)
    assert_array_almost_equal(cm.muc_scores()[:2], [1.0, 1.0], 4)

    # row 4 is exactly the same as row 1

    # row 5
    p1 = ["A B C".split()]
    p2 = ["A C".split(), "B"]
    cm = ClusteringMetrics.from_partitions(p1, p2)
    assert_array_almost_equal(cm.muc_scores()[:2], [1.0, 0.5], 4)


def test_IR_example():
    """Test example from IR book by Manning et al.

    The example gives 3 clusters and 17 points total. It is described on
    http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    """
    ltrue = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2)
    lpred = (0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 2, 2)
    cm = ClusteringMetrics.from_labels(ltrue, lpred)

    # test perfect variants
    rd = cm.row_diag()
    cd = cm.col_diag()
    assert_almost_equal(rd.assignment_score(model='m3'), 1.0, 6)
    assert_almost_equal(cd.assignment_score(model='m3'), 1.0, 6)
    assert_almost_equal(cd.assignment_score(model='m3', discrete=True),   1.0, 6)
    assert_almost_equal(rd.assignment_score(model='m3'), 1.0, 6)
    assert_almost_equal(rd.assignment_score(model='m3', discrete=True),   1.0, 6)

    # test that no redraws happen by default
    assert_almost_equal(cm.assignment_score(model='m3'),
                        cm.assignment_score(model='m3'), 6)

    ex = cm.expected(discrete=False)
    assert_almost_equal(ex.assignment_score(model='m3'), 0.0, 6)

    # test that H1 results in greater score than H0
    ex = cm.expected(discrete=True)
    assert_greater(
        cm.assignment_score(model='m3'),
        ex.assignment_score(model='m3'))

    # test entropy metrics
    h, c, v = cm.entropy_scores()
    assert_almost_equal(h, 0.371468, 6)
    assert_almost_equal(c, 0.357908, 6)
    assert_almost_equal(v, 0.364562, 6)

    assert_almost_equal(cm.vi_similarity(model=None),    0.517754, 6)
    assert_almost_equal(cm.vi_similarity(model='m1'),    0.378167, 6)
    assert_almost_equal(cm.vi_similarity(model='m2r'),   0.365605, 6)
    assert_almost_equal(cm.vi_similarity(model='m2c'),   0.377165, 6)
    assert_almost_equal(cm.vi_similarity(model='m3'),    0.364562, 6)

    assert_almost_equal(cm.mirkin_match_coeff(),         0.695502, 6)
    assert_almost_equal(cm.rand_index(),                 0.676471, 6)
    assert_almost_equal(cm.fowlkes_mallows(),            0.476731, 6)
    assert_almost_equal(cm.assignment_score(model=None), 0.705882, 6)
    assert_almost_equal(cm.assignment_score(model='m3'), 0.554974, 6)

    assert_almost_equal(cm.chisq_score(),          11.900000, 6)
    assert_almost_equal(cm.g_score(),              13.325845, 6)

    # test metrics that are based on pairwise co-association matrix
    conf = cm.pairwise

    assert_almost_equal(conf.chisq_score(),         8.063241, 6)
    assert_almost_equal(conf.g_score(),             7.804221, 6)

    assert_almost_equal(conf.jaccard_coeff(),       0.312500, 6)
    assert_almost_equal(conf.ochiai_coeff(),        0.476731, 6)
    assert_almost_equal(conf.dice_coeff(),          0.476190, 6)
    assert_almost_equal(conf.sokal_sneath_coeff(),  0.185185, 6)

    assert_almost_equal(conf.kappa(),               0.242915, 6)
    assert_almost_equal(conf.accuracy(),            0.676471, 6)
    assert_almost_equal(conf.precision(),           0.500000, 6)
    assert_almost_equal(conf.recall(),              0.454545, 6)

    exp_tw = _talburt_wang_index(ltrue, lpred)
    act_tw = cm.talburt_wang_index()
    assert_almost_equal(exp_tw, act_tw, 6)


def test_adjustment_for_chance():
    """Check that adjusted scores are almost zero on random labels
    """
    n_clusters_range = [2, 10, 50, 90]
    n_samples = 100
    n_runs = 10

    scores = uniform_labelings_scores(
        adjusted_rand_score, n_samples, n_clusters_range, n_runs)

    max_abs_scores = np.abs(scores).max(axis=1)
    assert_array_almost_equal(max_abs_scores, [0.02, 0.03, 0.03, 0.02], 2)


def test_twoway_confusion_1():
    """Finley's tornado data
    http://www.cawcr.gov.au/projects/verification/Finley/Finley_Tornados.html
    """
    cm = ConfusionMatrix2.from_ccw(28, 72, 2680, 23)

    assert_almost_equal(cm.g_score(),       126.1, 1)
    assert_almost_equal(cm.chisq_score(),   397.9, 1)

    mic0, mic1, mic2 = cm.mic_scores()
    assert_almost_equal(mic2,       0.429, 3)
    assert_almost_equal(mic1,       0.497, 3)
    assert_almost_equal(mic0,       0.382, 3)

    assert_almost_equal(cm.matthews_corr(), 0.377, 3)
    assert_almost_equal(cm.informedness(),  0.523, 3)
    assert_almost_equal(cm.markedness(),    0.271, 3)

    kappa0, kappa1, kappa2 = cm.kappas()
    assert_almost_equal(kappa0,        0.267, 3)
    assert_almost_equal(kappa1,        0.532, 3)
    assert_almost_equal(kappa2,        0.355, 3)


def test_twoway_confusion_2():
    """Finley's tornado data (listed in Goodman and Kruskal)
    """
    cm = ConfusionMatrix2.from_ccw(11, 14, 906, 3)

    assert_almost_equal(cm.g_score(),       70.83, 2)
    assert_almost_equal(cm.chisq_score(),   314.3, 1)

    mic0, mic1, mic2 = cm.mic_scores()
    assert_almost_equal(mic0,      0.555, 3)
    assert_almost_equal(mic1,      0.698, 3)
    assert_almost_equal(mic2,      0.614, 3)

    assert_almost_equal(cm.matthews_corr(), 0.580, 3)
    assert_almost_equal(cm.informedness(),  0.770, 3)
    assert_almost_equal(cm.markedness(),    0.437, 3)

    kappa0, kappa1, kappa2 = cm.kappas()
    assert_almost_equal(kappa0,        0.431, 3)
    assert_almost_equal(kappa1,        0.780, 3)
    assert_almost_equal(kappa2,        0.556, 3)


def test_negative_correlation():
    """Some metrics should have negative sign
    """
    cm = ConfusionMatrix2.from_ccw(10, 120, 8, 300)
    assert_almost_equal(cm.g_score(),        384.52, 2)
    assert_almost_equal(cm.chisq_score(),    355.70, 2)

    mic0, mic1, mic2 = cm.mic_scores()
    assert_almost_equal(mic0,      -0.8496, 4)
    assert_almost_equal(mic1,      -0.8524, 4)
    assert_almost_equal(mic2,      -0.8510, 4)

    assert_almost_equal(cm.matthews_corr(), -0.9012, 4)
    assert_almost_equal(cm.informedness(),  -0.9052, 4)
    assert_almost_equal(cm.markedness(),    -0.8971, 4)
    assert_almost_equal(cm.kappa(),         -0.6407, 4)
    inform, marked = cm.informedness(), cm.markedness()
    expected_matt = geometric_mean(inform, marked)
    assert_almost_equal(expected_matt, cm.matthews_corr(), 6)


def test_twoway_confusion_phi():
    cm = ConfusionMatrix2.from_ccw(116, 21, 18, 21)
    assert_almost_equal(cm.matthews_corr(), 0.31, 2)
    assert_almost_equal(cm.yule_q(), 0.6512, 4)
    assert_almost_equal(cm.DOR(),    4.7347, 4)

    cm = ConfusionMatrix2.from_ccw(35, 60, 41, 9)
    assert_almost_equal(cm.chisq_score(), 5.50, 2)


def test_kappa_precalculated():
    # from literature
    assert_almost_equal(cohen_kappa(22, 4, 11, 2),
                        0.67, 2)
    assert_almost_equal(product_moment(22, 4, 11, 2),
                        0.67, 2)
    assert_almost_equal(cohen_kappa(147, 10, 62, 3),
                        0.86, 2)
    assert_almost_equal(product_moment(147, 10, 62, 3),
                        0.87, 2)
    # numeric stability cases
    assert_almost_equal(cohen_kappa(69, 1, 3, 11),
                        0.280000, 6)
    assert_almost_equal(product_moment(69, 1, 3, 11),
                        0.350000, 6)
    assert_almost_equal(cohen_kappa(1, 2, 96, 5),
                        0.191111, 6)
    assert_almost_equal(product_moment(1, 2, 96, 5),
                        0.203746, 6)
