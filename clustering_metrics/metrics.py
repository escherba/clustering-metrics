# -*- coding: utf-8 -*-

"""
"""

import warnings
import numpy as np
from math import log, sqrt, copysign
from collections import Set, namedtuple
from pymaptools.containers import CrossTab, OrderedCrossTab
from pymaptools.iter import iter_items, isiterable
from pymaptools.sample import randround
from clustering_metrics.utils import _div, _log
from clustering_metrics.entropy import fentropy, fnum_pairs, fsum_pairs, \
    emi_from_margins, assignment_cost
from scipy.stats import fisher_exact


def jaccard_similarity(iterable1, iterable2):
    """Jaccard similarity between two sets

    Parameters
    ----------
    iterable1 : collections.Iterable
        first bag of items (order irrelevant)

    iterable2 : collections.Iterable
        second bag of items (order irrelevant)

    Returns
    -------

    jaccard_similarity : float
    """
    t = ConfusionMatrix2.from_sets(iterable1, iterable2)
    return t.jaccard_coeff()


def ratio2weights(ratio):
    """Numerically accurate conversion of ratio of two weights to weights
    """
    if ratio <= 1.0:
        lweight = ratio / (1.0 + ratio)
    else:
        lweight = 1.0 / (1.0 + 1.0 / ratio)
    return lweight, 1.0 - lweight


def geometric_mean(x, y):
    """Geometric mean of two numbers. Always returns a float

    Although geometric mean is defined for negative numbers, Scipy function
    doesn't allow it. Hence this function
    """
    prod = x * y
    if prod < 0.0:
        raise ValueError("x and y have different signs")
    return copysign(1, x) * sqrt(prod)


def geometric_mean_weighted(x, y, ratio=1.0):
    """Geometric mean of two numbers with a weight ratio. Returns a float

    ::

        >>> geometric_mean_weighted(1, 4, ratio=1.0)
        2.0
        >>> geometric_mean_weighted(1, 4, ratio=0.0)
        1.0
        >>> geometric_mean_weighted(1, 4, ratio=float('inf'))
        4.0
    """
    lweight, rweight = ratio2weights(ratio)
    lsign = copysign(1, x)
    rsign = copysign(1, y)
    if lsign != rsign and x != y:
        raise ValueError("x and y have different signs")
    return lsign * (abs(x) ** rweight) * (abs(y) ** lweight)


def unitsq_sigmoid(x, s=0.5):
    """Unit square sigmoid (for transforming P-like scales)

    ::

        >>> round(unitsq_sigmoid(0.1), 4)
        0.25
        >>> round(unitsq_sigmoid(0.9), 4)
        0.75

    """
    a = x ** s
    b = (1 - x) ** s
    return a / (a + b)


def harmonic_mean(x, y):
    """Harmonic mean of two numbers. Always returns a float
    """
    return float(x) if x == y else 2.0 * (x * y) / (x + y)


def harmonic_mean_weighted(x, y, ratio=1.0):
    """Harmonic mean of two numbers with a weight ratio. Returns a float

    ::

        >>> harmonic_mean_weighted(1, 3, ratio=1.0)
        1.5
        >>> harmonic_mean_weighted(1, 3, ratio=0.0)
        1.0
        >>> harmonic_mean_weighted(1, 3, ratio=float('inf'))
        3.0
    """
    lweight, rweight = ratio2weights(ratio)
    return float(x) if x == y else (x * y) / (lweight * x + rweight * y)


class ContingencyTable(CrossTab):

    # Note: not subclassing Pandas DataFrame because the goal is to specifically
    # optimize for sparse use cases when >90% of the table consists of zeros.
    # As of today, Pandas 'crosstab' implementation of frequency tables forces
    # one to iterate on all the zeros, which is horrible...

    def __init__(self, *args, **kwargs):
        CrossTab.__init__(self, *args, **kwargs)
        self._assignment_cost = None
        self._expected_freqs_ = {}
        self._expected_freqs_discrete_ = {}

    def to_array(self, default=0, cpad=False, rpad=False):
        """Convert to NumPy array
        """
        rows = self.to_rows(default=default, cpad=cpad, rpad=rpad)
        return np.array(rows)

    # Factory methods

    def expected_freqs_(self, model='m3'):
        table = self._expected_freqs_.get(model)
        if table is not None:
            return table

        N = float(self.grand_total)
        row_margin = self.row_totals
        col_margin = self.col_totals
        R, C = self.shape

        if model == 'm1':                # no fixed margin
            rows = self._row_type_2d()
            expected = N / float(len(row_margin) * len(col_margin))
            for (ri, ci), _ in self.iter_all():
                rows[ri][ci] = expected
        elif model == 'm2r':             # fixed row margin
            rows = self._row_type_2d()
            cm = N / float(len(col_margin))
            for (ri, ci), _ in self.iter_all():
                rm = row_margin[ri]
                numer = rm * cm
                if numer != 0:
                    rows[ri][ci] = numer / N
        elif model == 'm2c':             # fixed column margin
            rows = self._row_type_2d()
            rm = N / float(len(row_margin))
            for (ri, ci), _ in self.iter_all():
                cm = col_margin[ci]
                numer = rm * cm
                if numer != 0:
                    rows[ri][ci] = numer / N
        elif model == 'm3':              # fixed row *and* column margin
            rows = self._row_type_2d()
            for (ri, ci), _ in self.iter_all():
                rm = row_margin[ri]
                cm = col_margin[ci]
                numer = rm * cm
                if numer != 0:
                    rows[ri][ci] = numer / N
        elif model == 'x1':                # no fixed margin
            rows = np.zeros((R, C), dtype=int)
            rows[0][0] = self.grand_total
        elif model == 'x2r':             # fixed row margin
            rows = np.zeros((R, C), dtype=int)
            for idx, rm in enumerate(self.iter_row_totals()):
                rows[idx, 0] = rm
        elif model == 'x2c':             # fixed column margin
            rows = np.zeros((R, C), dtype=int)
            for idx, cm in enumerate(self.iter_col_totals()):
                rows[0, idx] = cm
        else:
            raise NotImplementedError(model)
        self._expected_freqs_[model] = table = self.__class__(rows=rows)
        return table

    def expected(self, model='m3', discrete=False, redraw=False):
        """Factory creating expected table given current margins
        """
        if discrete:
            table = self._expected_freqs_discrete_.get(model)
        else:
            table = self.expected_freqs_(model)

        # continuous
        if not discrete:
            if table is None:
                raise RuntimeError("line should be unreachable")
            return table

        # discrete
        if table is None or redraw:
            continuous = self.expected_freqs_(model)
            rows = self._row_type_2d()

            # create a sparse instance
            for (ri, ci), expected in continuous.items():
                expected = randround(expected)
                if expected != 0:
                    rows[ri][ci] = expected

            self._expected_freqs_discrete_[model] = table = self.__class__(rows=rows)
        return table

    @staticmethod
    def _normalize_measure(value, maximum=1.0, center=0.0):
        """Normalize to maximum with optional centering
        """
        if isiterable(value):
            value = np.asarray(value)
        if isiterable(center):
            center = np.asarray(center)
        if isiterable(maximum):
            maximum = np.asarray(maximum)
        return np.divide(value - center, maximum - center)

    def adjust_to_null(self, measure, model='m3', with_warnings=False):
        """Adjust a measure to null model

        The general formula for chance correction of an association measure
        :math:`M` is:

        .. math::

            M_{adj} = \\frac{M - E(M)}{M_{max} - E(M)},

        where :math:`M_{max}` is the maximum value a measure :math:`M` can
        achieve, and :math:`E(M)` is the expected value of :math:`M` under
        statistical independence given fixed table margins. In simple cases,
        the expected value of a measure is the same as the value of the measure
        given a null model. This is not always the case, however, and, to
        properly adjust for chance, sometimes one has average over all possible
        contingency tables using hypergeometric distribution for example.

        The method returns a tuple for two different measure ceilings: row-
        diagonal and column-diagonal. For symmetric measures, the two values
        will be the same.
        """
        if callable(measure):
            measure = measure.__name__
        actual = getattr(self, measure)()
        model = getattr(self.expected(model=model), measure)()
        if with_warnings and np.isclose(np.sum(model), 0.0):
            warnings.warn("'%s' is already centered" % measure)
        max_row = getattr(self.row_diag(), measure)()
        if with_warnings and np.isclose(np.average(max_row), 1.0):
            warnings.warn("'%s' is already row-normalized" % measure)
        max_col = getattr(self.col_diag(), measure)()
        if with_warnings and np.isclose(np.average(max_col), 1.0):
            warnings.warn("'%s' is already column-normalized" % measure)
        row_adjusted = self._normalize_measure(actual, max_row, model)
        col_adjusted = self._normalize_measure(actual, max_col, model)
        return row_adjusted, col_adjusted

    def row_diag(self):
        """Factory creating diagonal table given current row margin
        """
        rows = self._row_type_2d()
        for i, m in iter_items(self.row_totals):
            rows[i][i] = m
        return self.__class__(rows=rows)

    def col_diag(self):
        """Factory creating diagonal table given current column margin
        """
        rows = self._row_type_2d()
        for i, m in iter_items(self.col_totals):
            rows[i][i] = m
        return self.__class__(rows=rows)

    # Misc metrics
    def chisq_score(self):
        """Pearson's chi-square statistic

        >>> r = {1: {1: 16, 3: 2}, 2: {1: 1, 2: 3}, 3: {1: 4, 2: 5, 3: 5}}
        >>> cm = ContingencyTable(rows=r)
        >>> round(cm.chisq_score(), 3)
        19.256

        """
        N = float(self.grand_total)
        score = 0.0
        for rm, cm, observed in self.iter_all_with_margins():
            numer = rm * cm
            if numer != 0:
                expected = numer / N
                score += (observed - expected) ** 2 / expected
        return score

    def g_score(self):
        """G-statistic for RxC contingency table

        This method does not perform any corrections to this statistic (e.g.
        Williams', Yates' corrections).

        The statistic is equivalent to the negative of Mutual Information times
        two.  Mututal Information on a contingency table is defined as the
        difference between the information in the table and the information in
        an independent table with the same margins.  For application of mutual
        information (in the form of G-score) to search for collocated words in
        NLP, see [1]_ and [2]_.

        References
        ----------

        .. [1] `Dunning, T. (1993). Accurate methods for the statistics of
               surprise and coincidence. Computational linguistics, 19(1), 61-74.
               <http://dl.acm.org/citation.cfm?id=972454>`_

        .. [2] `Ted Dunning's personal blog entry and the discussion under it.
               <http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html>`_

        """
        _, _, I_CK = self._entropies()
        return 2.0 * I_CK

    def _entropies(self):
        """Return H_C, H_K, and mutual information

        Not normalized by N
        """
        H_C = fentropy(self.row_totals)
        H_K = fentropy(self.col_totals)
        H_actual = fentropy(self.values())
        H_expected = H_C + H_K
        I_CK = H_expected - H_actual
        return H_C, H_K, I_CK

    def mutual_info_score(self):
        """Mutual Information Score

        Mutual Information (divided by N).

        The metric is equal to the Kullback-Leibler divergence of the joint
        distribution with the product distribution of the marginals.
        """
        _, _, I_CK = self._entropies()
        return I_CK / self.grand_total

    def entropy_scores(self, mean='harmonic'):
        """Gives three entropy-based metrics for a RxC table

        The metrics are: Homogeneity, Completeness, and V-measure

        The V-measure metric is also known as Normalized Mutual Information
        (NMI), and is calculated here as the harmonic mean of Homogeneity and
        Completeness (:math:`NMI_{sum}`). There exist other definitions of NMI (see
        Table 2 in [1]_ for a good review).

        Homogeneity and Completeness are duals of each other and can be thought
        of (although this is not technically accurate) as squared regression
        coefficients of a given clustering vs true labels (homogeneity) and of
        the dual problem of true labels vs given clustering (completeness).
        Because of the dual property, in a symmetric matrix, all three scores
        are the same. Homogeneity has an overall profile similar to that of
        precision in information retrieval. Completeness roughly corresponds to
        recall.

        This method replaces ``homogeneity_completeness_v_measure`` method in
        Scikit-Learn.  The Scikit-Learn version takes up :math:`O(n^2)` space
        because it stores data in a dense NumPy array, while the given version
        is sub-quadratic because of sparse underlying storage.

        Note that the entropy variables H in the code below are improperly
        defined because they ought to be divided by N (the grand total for the
        contingency table). However, the N variable cancels out during
        normalization.

        References
        ----------

        .. [1] `Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic
               measures for clusterings comparison: Variants, properties,
               normalization and correction for chance. The Journal of Machine
               Learning Research, 11, 2837-2854.
               <http://www.jmlr.org/papers/v11/vinh10a.html>`_

        """
        # ensure non-negative values by taking max of 0 and given value
        H_C, H_K, I_CK = self._entropies()
        h = 1.0 if H_C == 0.0 else max(0.0, I_CK / H_C)
        c = 1.0 if H_K == 0.0 else max(0.0, I_CK / H_K)
        if mean == 'harmonic':
            rsquare = harmonic_mean(h, c)
        elif mean == 'geometric':
            rsquare = geometric_mean(h, c)
        else:
            raise NotImplementedError(mean)
        return h, c, rsquare

    def adjusted_mutual_info(self):
        """Adjusted Mutual Information for two partitions

        For a mathematical definition, see [1]_, [2]_, and [2]_.

        References
        ----------

        .. [1] `Vinh, N. X., Epps, J., & Bailey, J. (2009, June). Information
               theoretic measures for clusterings comparison: is a correction
               for chance necessary?. In Proceedings of the 26th Annual
               International Conference on Machine Learning (pp. 1073-1080).
               ACM.
               <https://doi.org/10.1145/1553374.1553511>`_

        .. [2] `Vinh, N. X., & Epps, J. (2009, June). A novel approach for
               automatic number of clusters detection in microarray data based
               on consensus clustering. In Bioinformatics and BioEngineering,
               2009.  BIBE'09. Ninth IEEE International Conference on (pp.
               84-91). IEEE.
               <http://dx.doi.org/10.1109/BIBE.2009.19>`_

        .. [3] `Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic
               measures for clusterings comparison: Variants, properties,
               normalization and correction for chance. The Journal of Machine
               Learning Research, 11, 2837-2854.
               <http://www.jmlr.org/papers/v11/vinh10a.html>`_

        """
        # Prepare row totals and check for special cases
        row_totals = np.fromiter(self.iter_row_totals(), dtype=np.int64)
        col_totals = np.fromiter(self.iter_col_totals(), dtype=np.int64)
        R = len(row_totals)
        C = len(col_totals)
        if R == C == 1 or R == C == 0:
            # No clustering since the data is not split. This is a perfect match
            # hence return 1.0.
            return 1.0

        # In one step, calculate entropy for each labeling and mutual
        # information
        h_true, h_pred, mi = self._entropies()
        mi_max = max(h_true, h_pred)

        # Calculate the expected value for the MI
        emi = emi_from_margins(row_totals, col_totals)

        # Calculate the adjusted MI score
        ami = (mi - emi) / (mi_max - emi)
        return ami

    def assignment_score_m1(self, normalize=True, redraw=False):
        return self.assignment_score(
            normalize=normalize, model='m1', discrete=False, redraw=redraw)

    def assignment_score_m2r(self, normalize=True, redraw=False):
        return self.assignment_score(
            normalize=normalize, model='m2r', discrete=False, redraw=redraw)

    def assignment_score_m2c(self, normalize=True, redraw=False):
        return self.assignment_score(
            normalize=normalize, model='m2c', discrete=False, redraw=redraw)

    def assignment_score_m3(self, normalize=True, redraw=False):
        return self.assignment_score(
            normalize=normalize, model='m3', discrete=False, redraw=redraw)

    def assignment_score(self, normalize=True, model='m1',
                         discrete=False, redraw=False):
        """Similarity score by solving the Linear Sum Assignment Problem

        This metric is uniformly more powerful than the similarly behaved
        ``split_join_similarity`` which relies on an approximation to the
        optimal solution evaluated here. The split-join approximation
        asymptotically approaches the optimal solution as the clustering
        quality improves.

        On the ``model`` parameter: adjusting assignment cost for chance
        by relying on the hypergeometric distribution is extremely
        computationally expensive, but one way to get a better behaved metric
        is to just subtract the cost of a null model from the obtained score
        (in case of normalization, the null cost also has to be subtracted from
        the maximum cost). Note that on large tables even finding the null cost
        is too expensive, since expected tables have a lot less sparsity. Hence
        the parameter is off by default.

        Alternatively this problem can be recast as that of finding a *maximum
        weighted bipartite match* [1]_.

        This method of partition comparison was first mentioned in [2]_, given
        an approximation in [3]_, formally elaborated in [4]_ and empirically
        compared with other measures in [5]_.

        See Also
        --------
        split_join_similarity

        References
        ----------

        .. [1] `Wikipedia entry on weighted bipartite graph matching
               <https://en.wikipedia.org/wiki/Matching_%28graph_theory%28#In_weighted_bipartite_graphs>`_

        .. [2] `Almudevar, A., & Field, C. (1999). Estimation of
               single-generation sibling relationships based on DNA markers.
               Journal of agricultural, biological, and environmental
               statistics, 136-165.
               <http://www.jstor.org/stable/1400594>`_

        .. [3] `Ben-Hur, A., & Guyon, I. (2003). Detecting stable clusters
               using principal component analysis. In Functional Genomics (pp.
               159-182). Humana press.
               <http://doi.org/10.1385/1-59259-364-X:159>`_

        .. [4] `Gusfield, D. (2002). Partition-distance: A problem and class of
               perfect graphs arising in clustering. Information Processing
               Letters, 82(3), 159-164.
               <http://doi.org/10.1016/S0020-0190%2801%2900263-0>`_

        .. [5] `Giurcaneanu, C. D., & Tabus, I. (2004). Cluster structure
               inference based on clustering stability with applications to
               microarray data analysis. EURASIP Journal on Applied Signal
               Processing, 2004, 64-80.
               <http://dl.acm.org/citation.cfm?id=1289345>`_

        """

        # computing assignment cost is expensive so we cache it
        cost = self._assignment_cost

        if cost is None:
            # guess matrix dtype
            cost = assignment_cost(self.to_rows(), maximize=True)
            self._assignment_cost = cost

        N = self.grand_total
        R, C = self.shape

        if model is None:
            null_cost = 0
        elif (not discrete) and model == 'm1':
            # No margin is fixed, assignment doesn't matter (all cells are
            # equal under this assumption), so we can calculate expected cost
            # directly
            null_cost = N / float(max(R, C))
        elif (not discrete) and model == 'm2r':
            # fixed row margin, assignment also doesn't matter
            sum_top_rows = N if R <= C else \
                sum(sorted(self.row_totals.values(), reverse=True)[:C])
            null_cost = sum_top_rows / float(C)
        elif (not discrete) and model == 'm2c':
            # fixed column margin, assignment also doesn't matter
            sum_top_cols = N if C <= R else \
                sum(sorted(self.col_totals.values(), reverse=True)[:R])
            null_cost = sum_top_cols / float(R)
        else:
            # all margins fixed, assignment matters
            expected = self.expected(
                model=model, discrete=discrete, redraw=redraw)
            null_cost = expected.assignment_score(
                model=None, normalize=False)

        cost -= null_cost
        if normalize:
            max_cost = N - null_cost
            cost = 1.0 if cost == max_cost else _div(cost, max_cost)

        return cost

    def vi_distance(self, normalize=True):
        """Variation of Information distance

        Defined in [1]_. This is one of several possible entropy-based distance
        measures that could be defined on a RxC matrix. The given measure is
        equivalent to :math:`2 D_{sum}` as listed in Table 2 in [2]_.

        Note that the entropy variables H below are calculated using natural
        logs, so a base correction may be necessary if you need your result in
        base 2 for example.

        References
        ----------

        .. [1] `Meila, M. (2007). Comparing clusterings -- an information based
               distance. Journal of multivariate analysis, 98(5), 873-895.
               <https://doi.org/10.1016/j.jmva.2006.11.013>`_

        .. [2] `Vinh, N. X., Epps, J., & Bailey, J. (2010). Information theoretic
               measures for clusterings comparison: Variants, properties,
               normalization and correction for chance. The Journal of Machine
               Learning Research, 11, 2837-2854.
               <http://www.jmlr.org/papers/v11/vinh10a.html>`_

        """
        H_C, H_K, I_CK = self._entropies()
        N = self.grand_total
        score = (H_C + H_K - 2 * I_CK) / N
        if normalize:
            score = _div(score, log(N))
        return score

    def vi_similarity_m1(self, normalize=True):
        return self.vi_similarity(normalize=normalize, model='m1')

    def vi_similarity_m2r(self, normalize=True):
        return self.vi_similarity(normalize=normalize, model='m2r')

    def vi_similarity_m2c(self, normalize=True):
        return self.vi_similarity(normalize=normalize, model='m2c')

    def vi_similarity_m3(self, normalize=True):
        return self.vi_similarity(normalize=normalize, model='m3')

    def vi_similarity(self, normalize=True, model='m1'):
        """Inverse of ``vi_distance``

        The m1 adjustment is monotonic for tables of fixed size. The m3
        adjustment turns this measure into Normalized Mutual Information (NMI)
        """
        R, C = self.shape
        N = self.grand_total

        max_dist = log(N)
        dist = self.vi_distance(normalize=False)
        score = max_dist - dist

        if model is None:
            null_score = 0
        elif model == 'm1':         # only N is fixed
            null_dist = log(R) + log(C)
            null_score = max_dist - null_dist
        elif model == 'm2r':        # fixed row margin
            null_dist = log(C) + fentropy(self.row_totals) / N
            null_score = max_dist - null_dist
        elif model == 'm2c':        # fixed column margin
            null_dist = log(R) + fentropy(self.col_totals) / N
            null_score = max_dist - null_dist
        elif model == 'm3':         # both row and column margins fixed
            null_dist = (fentropy(self.row_totals) + fentropy(self.col_totals)) / N
            null_score = max_dist - null_dist
        else:
            expected = self.expected(model)
            null_score = expected.vi_similarity(normalize=False, model=None)

        score -= null_score
        if normalize:
            max_score = max_dist - null_score
            score = 1.0 if score == max_score else _div(score, max_score)

        return score

    def split_join_distance(self, normalize=True):
        """Distance metric based on ``split_join_similarity``
        """
        sim = self.split_join_similarity(normalize=False, model=None)
        max_sim = 2 * self.grand_total
        score = max_sim - sim
        if normalize:
            score = _div(score, max_sim)
        return score

    def split_join_similarity_m1(self, normalize=True):
        return self.split_join_similarity(normalize=normalize, model='m1')

    def split_join_similarity_m2r(self, normalize=True):
        return self.split_join_similarity(normalize=normalize, model='m2r')

    def split_join_similarity_m2c(self, normalize=True):
        return self.split_join_similarity(normalize=normalize, model='m2c')

    def split_join_similarity_m3(self, normalize=True):
        return self.split_join_similarity(normalize=normalize, model='m3')

    def split_join_similarity(self, normalize=True, model='m1'):
        """Split-join similarity score

        Split-join similarity is a two-way assignment-based score first
        proposed in [1]_. The distance variant of this measure has metric
        properties.  Like the better known purity score (a one-way
        coefficient), this measure implicitly performs class-cluster
        assignment, except the assignment is performed twice: based on the
        corresponding maximum frequency in the contingency table, each class is
        given a cluster with the assignment weighted according to the
        frequency, then the procedure is inversed to assign a class to each
        cluster. The final unnormalized distance score comprises of a simple
        sum of the two one-way assignment scores.

        By default, ``m1`` null model is subtracted, to make the final
        score independent of the number of clusters::

            >>> t2 = ClusteringMetrics(rows=10 * np.ones((2, 2), dtype=int))
            >>> t2.split_join_similarity(model=None)
            0.5
            >>> t2.split_join_similarity(model='m1')
            0.0
            >>> t8 = ClusteringMetrics(rows=10 * np.ones((8, 8), dtype=int))
            >>> t8.split_join_similarity(model=None)
            0.125
            >>> t8.split_join_similarity(model='m1')
            0.0

        See Also
        --------
        assignment_score

        References
        ----------

        .. [1] `Dongen, S. V. (2000). Performance criteria for graph clustering
               and Markov cluster experiments. Information Systems [INS],
               (R 0012), 1-36.
               <http://dl.acm.org/citation.cfm?id=868979>`_

        """
        pa_B = sum(max(row) for row in self.iter_rows())
        pb_A = sum(max(col) for col in self.iter_cols())
        score = pa_B + pb_A

        N = self.grand_total
        R, C = self.shape

        if model is None:
            null_score = 0
        elif model == 'm1':         # only N is fixed
            null_score = N / float(R) + N / float(C)
        elif model == 'm2r':        # fixed row margin
            null_score = max(self.row_totals.values()) + N / float(C)
        elif model == 'm2c':        # fixed column margin
            null_score = N / float(R) + max(self.col_totals.values())
        elif model == 'm3':         # both row and column margins fixed
            null_score = \
                max(self.row_totals.values()) + \
                max(self.col_totals.values())
        else:
            expected = self.expected(model)
            null_score = expected.split_join_similarity(normalize=False, model=None)

        score -= null_score
        if normalize:
            max_score = 2 * N - null_score
            score = 1.0 if score == max_score else _div(score, max_score)

        return score

    def talburt_wang_index(self):
        """Talburt-Wang index of similarity of two partitions

        On sparse matrices, the resolving power of this measure asymptotically
        approaches that of assignment-based scores such as ``assignment_score``
        and ``split_join_similarity``, however on dense matrices this measure
        will not perform well due to its reliance on category cardinalities
        (how many types were seen) rather than on observation counts (how many
        instances of each type were seen).

        A relatively decent clustering::

            >>> a = [ 1,  1,  1,  2,  2,  2,  2,  3,  3,  4]
            >>> b = [43, 56, 56,  5, 36, 36, 36, 74, 74, 66]
            >>> t = ContingencyTable.from_labels(a, b)
            >>> round(t.talburt_wang_index(), 3)
            0.816

        Less good clustering (example from [1]_)::

            >>> clusters = [[1, 1], [1, 1, 1, 1], [2, 3], [2, 2, 3, 3],
            ...             [3, 3, 4], [3, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
            >>> t = ContingencyTable.from_clusters(clusters)
            >>> round(t.talburt_wang_index(), 2)
            0.49

        References
        ----------

        .. [1] `Talburt, J., Wang, R., Hess, K., & Kuo, E. (2007). An algebraic
               approach to data quality metrics for entity resolution over large
               datasets.  Information quality management: Theory and
               applications, 1-22.
               <http://www.igi-global.com/chapter/algebraic-approach-data-quality-metrics/23022>`_
        """
        A_card, B_card = self.shape
        V_card = len(self)
        return _div(sqrt(A_card * B_card), V_card)

    def muc_scores(self):
        """MUC similarity indices for coreference scoring

        Implemented after description in [1]_. The compound fscore-like metric
        has good resolving power on sparse models, similar to
        ``fowlkes_mallows`` (pairwise ``ochiai_coeff``), however it becomes
        useless on dense matrices as it relies on category cardinalities (how
        many types were seen) rather than on observation counts (how many
        instances of each type were seen).

        ::

            >>> p1 = [x.split() for x in ["A B C", "D E F G"]]
            >>> p2 = [x.split() for x in ["A B", "C", "D", "E", "F G"]]
            >>> cm = ClusteringMetrics.from_partitions(p1, p2)
            >>> cm.muc_scores()[:2]
            (1.0, 0.4)

        Elements that are part of neither partition (in this case, E) are
        excluded from consideration::

            >>> p1 = [x.split() for x in ["A B", "C", "D", "F G", "H"]]
            >>> p2 = [x.split() for x in ["A B", "C D", "F G H"]]
            >>> cm = ClusteringMetrics.from_partitions(p1, p2)
            >>> cm.muc_scores()[:2]
            (0.5, 1.0)

        References
        ----------

        .. [1] `Vilain, M., Burger, J., Aberdeen, J., Connolly, D., &
               Hirschman, L. (1995, November). A model-theoretic coreference
               scoring scheme. In Proceedings of the 6th conference on Message
               understanding (pp. 45-52).  Association for Computational
               Linguistics.
               <http://www.aclweb.org/anthology/M/M95/M95-1005.pdf>`_
        """
        A_card, B_card = self.shape
        V_card = len(self)
        N = self.grand_total

        recall = _div(N - V_card,  N - A_card)
        precision = _div(N - V_card,  N - B_card)
        fscore = harmonic_mean(recall, precision)
        return precision, recall, fscore

    def bc_metrics(self):
        """'B-cubed' precision, recall, and fscore

        As described in [1]_ and [2]_. Was extended to overlapping clusters in
        [3]_.  These metrics perform very similarly to normalized entropy
        metrics (homogeneity, completeness, V-measure).

        References
        ----------

        .. [1] `Bagga, A., & Baldwin, B. (1998, August). Entity-based cross-
               document coreferencing using the vector space model. In
               Proceedings of the 36th Annual Meeting of the Association for
               Computational Linguistics and 17th International Conference on
               Computational Linguistics-Volume 1 (pp. 79-85).  Association for
               Computational Linguistics.
               <https://aclweb.org/anthology/P/P98/P98-1012.pdf>`_

        .. [2] `Bagga, A., & Baldwin, B. (1998, May). Algorithms for scoring
               coreference chains. In The first international conference on
               language resources and evaluation workshop on linguistics
               coreference (Vol. 1, pp. 563-566).
               <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.47.5848>`_

        .. [3] `Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009). A
               comparison of extrinsic clustering evaluation metrics based on
               formal constraints. Information retrieval, 12(4), 461-486.
               <http://doi.org/10.1007/s10791-008-9066-8>`_
        """
        precision = 0.0
        recall = 0.0
        for rm, cm, observed in self.iter_vals_with_margins():
            precision += (observed ** 2) / float(cm)
            recall += (observed ** 2) / float(rm)
        N = self.grand_total
        precision = _div(precision, N)
        recall = _div(recall, N)
        fscore = harmonic_mean(recall, precision)
        return precision, recall, fscore


class ClusteringMetrics(ContingencyTable):

    """Provides external clustering evaluation metrics

    A subclass of ContingencyTable that builds a pairwise co-association matrix
    for clustering comparisons.

    ::

        >>> Y1 = {(1, 2, 3), (4, 5, 6)}
        >>> Y2 = {(1, 2), (3, 4, 5), (6,)}
        >>> cm = ClusteringMetrics.from_partitions(Y1, Y2)
        >>> cm.split_join_similarity(model=None)
        0.75
    """

    def __init__(self, *args, **kwargs):
        ContingencyTable.__init__(self, *args, **kwargs)
        self._pairwise = None
        self._pairwise_models = {}

    @property
    def pairwise(self):
        """Confusion matrix on all pair assignments from two partitions

        A partition of N is a set of disjoint clusters s.t. every point in N
        belongs to one and only one cluster and every cluster consists of at
        least one point. Given two partitions A and B and a co-occurrence
        matrix of point pairs,

        == =============================================================
        TP count of pairs found in the same partition in both A and B
        FP count of pairs found in the same partition in A but not in B
        FN count of pairs found in the same partition in B but not in A
        TN count of pairs in different partitions in both A and B
        == =============================================================

        Note that although the resulting confusion matrix has the form of a
        correlation table for two binary variables, it is not symmetric if the
        original partitions are not symmetric.

        """
        pairwise = self._pairwise
        if pairwise is None:
            actual_positives = fsum_pairs(self.iter_row_totals())
            called_positives = fsum_pairs(self.iter_col_totals())
            TP = fsum_pairs(self.values())
            FN = actual_positives - TP
            FP = called_positives - TP
            TN = fnum_pairs(self.grand_total) - TP - FP - FN
            pairwise = self._pairwise = ConfusionMatrix2.from_ccw(TP, FP, TN, FN)
        return pairwise

    def get_score(self, scoring_method, *args, **kwargs):
        """Evaluate specified scoring method
        """
        try:
            method = getattr(self, scoring_method)
        except AttributeError:
            method = getattr(self.pairwise, scoring_method)
        return method(*args, **kwargs)

    def adjusted_rand_index(self):
        """Rand score (accuracy) corrected for chance

        This is a memory-efficient replacement for a similar Scikit-Learn
        function.
        """
        return self.pairwise.kappa()

    def rand_index(self):
        """Pairwise accuracy (uncorrected for chance)

        Don't use this metric; it is only added here as the "worst reference"
        """
        return self.pairwise.accuracy()

    def fowlkes_mallows(self):
        """Fowlkes-Mallows index for partition comparison

        Defined as the Ochiai coefficient on the pairwise matrix
        """
        return self.pairwise.ochiai_coeff()

    def adjusted_fowlkes_mallows(self):
        """Fowlkes-Mallows index adjusted for chance

        Adjustmend for chance done by subtracting the expected (Model 3)
        pairwise matrix from the actual one. This coefficient appears to be
        uniformly more powerful than the unadjusted version. Compared to ARI
        and product-moment correlation coefficients, this index is generally
        less powerful except in particularly poorly specified cases, e.g.
        clusters of unequal size sampled with high error rate from a large
        population.
        """
        return self.pairwise.ochiai_coeff_adj()

    def mirkin_match_coeff(self, normalize=True):
        """Equivalence match (similarity) coefficient

        Derivation of distance variant described in [1]_. This measure is
        nearly identical to pairwise unadjusted Rand index, as can be seen from
        the definition (Mirkin match formula uses square while pairwise
        accuracy uses n choose 2).

        ::

            >>> C3 = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}]
            >>> C4 = [{1, 2, 3, 4}, {5, 6, 7, 8, 9, 10, 11, 12}, {13, 14, 15, 16}]
            >>> t = ClusteringMetrics.from_partitions(C3, C4)
            >>> t.mirkin_match_coeff(normalize=False)
            216.0

        References
        ----------

        .. [1] `Mirkin, B (1996). Mathematical Classification and Clustering.
               Kluwer Academic Press: Boston-Dordrecht.
               <http://www.amazon.com/dp/0792341597>`_

        """
        max_score = self.grand_total ** 2
        score = max_score - self.mirkin_mismatch_coeff(normalize=False)
        if normalize:
            score = _div(score, max_score)
        return score

    def mirkin_mismatch_coeff(self, normalize=True):
        """Equivalence mismatch (distance) coefficient

        Direct formulation (without the pairwise abstraction):

        .. math::

            M = \\sum_{i=1}^{R} r_{i}^2 + \\sum_{j=1}^{C} c_{j}^2 - \\sum_{i=1}^{R}\\sum_{j=1}^{C} n_{ij}^2,

        where :math:`r` and :math:`c` are row and column margins, respectively,
        with :math:`R` and :math:`C` cardinalities.

        ::

            >>> C1 = [{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}]
            >>> C2 = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {11, 12, 13, 14, 15, 16}]
            >>> t = ClusteringMetrics.from_partitions(C1, C2)
            >>> t.mirkin_mismatch_coeff(normalize=False)
            56.0

        """
        score = 2 * (self.pairwise.FN + self.pairwise.FP)
        if normalize:
            score = _div(score, self.grand_total ** 2)
        return score


confmat2_type = namedtuple("ConfusionMatrix2", "TP FP TN FN")


class ConfusionMatrix2(ContingencyTable, OrderedCrossTab):
    """A confusion matrix (2x2 contingency table)

    For a binary variable (where one is measuring either presence vs absence of
    a particular feature), a confusion matrix where the ground truth levels are
    rows looks like this::

        >>> cm = ConfusionMatrix2(TP=20, FN=31, FP=14, TN=156)
        >>> cm
        ConfusionMatrix2(rows=[[20, 31], [14, 156]])
        >>> cm.to_array()
        array([[ 20,  31],
               [ 14, 156]])

    For a nominal variable, the negative class becomes a distinct label, and
    TP/FP/FN/TN terminology does not apply, although the algorithms should work
    the same way (with the obvious distinction that different assumptions will
    be made). For a convenient reference about some of the attributes and
    methods defined here see [1]_.

    Attributes
    ----------

    TP :
        True positive count
    FP :
        False positive count
    TN :
        True negative count
    FN :
        False negative count

    References
    ----------

    .. [1] `Wikipedia entry for Confusion Matrix
            <https://en.wikipedia.org/wiki/Confusion_matrix>`_
    """

    def __repr__(self):
        return "ConfusionMatrix2(rows=%s)" % repr(self.to_rows())

    def __init__(self, TP=None, FN=None, FP=None, TN=None, rows=None):
        if rows is None:
            rows = ((TP, FN), (FP, TN))
        ContingencyTable.__init__(self, rows=rows)

    def lform(self):
        """Factory creating L-form version of current table
        """
        a, c, d, b = self.to_ccw()
        if b < c:
            a += b
            b -= b
            c -= b
            d += b
        else:
            a += c
            b -= c
            c -= c
            d += c
        return self.__class__.from_ccw(a, c, d, b)

    @classmethod
    def from_sets(cls, set1, set2, universe_size=None):
        """Instantiate from two sets

        Accepts an optional universe_size parameter which allows us to take into
        account TN class and use probability-based similarity metrics.  Most of
        the time, however, set comparisons are performed ignoring this parameter
        and relying instead on non-probabilistic indices such as Jaccard's or
        Dice.
        """
        if not isinstance(set1, Set):
            set1 = set(set1)
        if not isinstance(set2, Set):
            set2 = set(set2)
        TP = len(set1 & set2)
        FP = len(set2) - TP
        FN = len(set1) - TP
        if universe_size is None:
            TN = 0
        else:
            TN = universe_size - TP - FP - FN
            if TN < 0:
                raise ValueError(
                    "universe_size must be at least as large as set union")
        return cls(TP, FN, FP, TN)

    @classmethod
    def from_random_counts(cls, low=0, high=100):
        """Instantiate from random values
        """
        return cls(*np.random.randint(low=low, high=high, size=(4,)))

    @classmethod
    def from_ccw(cls, TP, FP, TN, FN):
        """Instantiate from counter-clockwise form of TP FP TN FN
        """
        return cls(TP, FN, FP, TN)

    def to_ccw(self):
        """Convert to counter-clockwise form of TP FP TN FN
        """
        return confmat2_type(TP=self.TP, FP=self.FP, TN=self.TN, FN=self.FN)

    def get_score(self, scoring_method, *args, **kwargs):
        """Evaluate specified scoring method
        """
        method = getattr(self, scoring_method)
        return method(*args, **kwargs)

    @property
    def TP(self):
        return self.rows[0][0]

    @property
    def FN(self):
        return self.rows[0][1]

    @property
    def FP(self):
        return self.rows[1][0]

    @property
    def TN(self):
        return self.rows[1][1]

    def hypergeometric(self):
        """Hypergeometric association score
        """
        covsign = copysign(1, self.covar())
        _, pvalue = fisher_exact(self.to_array())
        return covsign * (-_log(pvalue))

    def ACC(self):
        """Accuracy (Rand Index)

        Synonyms: Simple Matching Coefficient, Rand Index
        """
        return _div(self.TP + self.TN, self.grand_total)

    def PPV(self):
        """Positive Predictive Value (Precision)

        Synonyms: precision, frequency of hits, post agreement, success ratio,
        correct alarm ratio
        """
        return _div(self.TP, self.TP + self.FP)

    def NPV(self):
        """Negative Predictive Value

        Synonyms: frequency of correct null forecasts
        """
        return _div(self.TN, self.TN + self.FN)

    def TPR(self):
        """True Positive Rate (Recall, Sensitivity)

        Synonyms: recall, sensitivity, hit rate, probability of detection,
        prefigurance
        """
        return _div(self.TP, self.TP + self.FN)

    def FPR(self):
        """False Positive Rate

        Synonyms: fallout
        """
        return _div(self.FP, self.TN + self.FP)

    def TNR(self):
        """True Negative Rate (Specificity)

        Synonyms: specificity
        """
        return _div(self.TN, self.FP + self.TN)

    def FNR(self):
        """False Negative Rate

        Synonyms: miss rate, frequency of misses
        """
        return _div(self.FN, self.TP + self.FN)

    def FDR(self):
        """False discovery rate

        Synonyms: false alarm ratio, probability of false alarm
        """
        return _div(self.FP, self.TP + self.FP)

    def FOR(self):
        """False omission rate

        Synonyms: detection failure ratio, miss ratio
        """
        return _div(self.FN, self.TN + self.FN)

    def PLL(self):
        """Positive likelihood ratio
        """
        return _div(self.TPR(), self.FPR())

    def NLL(self):
        """Negative likelihood ratio
        """
        return _div(self.FNR(), self.TNR())

    def DOR(self):
        """Diagnostics odds ratio

        Defined as

        .. math::

            DOR = \\frac{PLL}{NLL}.

        Odds ratio has a number of interesting/desirable properties, however
        its one peculiarity that leaves us looking for an alternative measure
        is that on L-shaped matrices like,

        .. math::

            \\begin{matrix} 77 & 0 \\\\ 5 & 26 \\end{matrix}

        its value will be infinity.

        Also known as: crude odds ratio, Mantel-Haenszel estimate.
        """
        a, c, d, b = self.to_ccw()
        ad, bc = a * d, b * c
        return _div(ad, bc)

    def ETS(self):
        """Equitable Threat Score
        """
        random_hits = _div((self.TP + self.FN) * (self.TP + self.FP), self.grand_total)
        numer = self.TP - random_hits
        denom = self.TP + self.FP + self.FN - random_hits
        return _div(numer, denom)

    def fscore(self, beta=1.0):
        """F-score

        As beta tends to infinity, F-score will approach recall.  As beta tends
        to zero, F-score will approach precision. A similarity coefficient that
        uses a similar definition is called Dice coefficient.

        See Also
        --------
        dice_coeff
        """
        return harmonic_mean_weighted(self.precision(), self.recall(), beta ** 2)

    def dice_coeff(self):
        """Dice similarity (Nei-Li coefficient)

        This is the same as F1-score, but calculated slightly differently here.
        Note that Dice can be zero if total number of positives is zero, but
        F-score is undefined in that case (because recall is undefined).

        When adjusted for chance, this coefficient becomes identical to
        ``kappa`` [1]_.

        Since this coefficient is monotonic with respect to Jaccard and Sokal
        Sneath coefficients, its resolving power is identical to that of the
        other two.

        See Also
        --------
        jaccard_coeff, sokal_sneath_coeff

        References
        ----------

        .. [1] `Albatineh, A. N., Niewiadomska-Bugaj, M., & Mihalko, D. (2006).
               On similarity indices and correction for chance agreement.
               Journal of Classification, 23(2), 301-313.
               <http://doi.org/10.1007/s00357-006-0017-z>`_
        """
        a, c, _, b = self.to_ccw()
        return _div(2 * a, 2 * a + b + c)

    def overlap_coeff(self):
        """Overlap coefficient (Szymkiewicz-Simpson coefficient)

        Can be obtained by standardizing Dice or Ochiai coefficients by their
        maximum possible value given fixed marginals. Not corrected for chance.

        Note that :math:`min(p_1, p_2)` is equal to the maximum value of
        :math:`a` given fixed marginals.

        When adjusted for chance, this coefficient turns into Loevinger's H.

        See Also
        --------
        loevinger_coeff
        """
        a, c, _, b = self.to_ccw()
        p1, p2 = a + b, a + c
        a_max = min(p1, p2)
        return 0.0 if a_max == 0 else _div(a, a_max)

    def jaccard_coeff(self):
        """Jaccard similarity coefficient

        Jaccard coefficient has an interesting property in that in L-shaped
        matrices where either FP or FN are close to zero, its scale becomes
        equivalent to the scale of either recall or precision respectively.

        Since this coefficient is monotonic with respect to Dice (F-score) and
        Sokal Sneath coefficients, its resolving power is identical to that of
        the other two.

        Jaccard index does not belong to the L-family of association indices
        and thus cannot be adjusted for chance by subtracting the its value
        under fixed-margin null model. Instead, its expectation must be
        calculated, for which no analytical solution exists [1]_.

        Synonyms: critical success index, threat score

        See Also
        --------
        dice_coeff, sokal_sneath_coeff

        References
        ----------

        .. [1] `Albatineh, A. N., & Niewiadomska-Bugaj, M. (2011). Correcting
               Jaccard and other similarity indices for chance agreement in
               cluster analysis. Advances in Data Analysis and Classification,
               5(3), 179-200.
               <http://doi.org/10.1007/s11634-011-0090-y>`_
        """
        a, c, _, b = self.to_ccw()
        return _div(a, a + b + c)

    def ochiai_coeff_adj(self):
        """Ochiai coefficient adjusted for chance

        This index is nearly identical to Mattthews' Correlation Coefficient,
        which should be used instead.

        See Also
        --------
        matthews_corr, ochiai_coeff
        """
        a, c, d, b = self.to_ccw()
        p1, p2 = a + b, a + c
        n = a + b + c + d

        if n == 0:
            return np.nan
        elif a == n or d == n:
            # only one (diagonal) cell is non-zero
            return 0.5
        elif p1 == 0 or p2 == 0:
            # first row or column is zero, second non-zero
            return 0.0

        p1_p2 = p1 * p2
        numer = n * a - p1_p2
        denom = n * sqrt(p1_p2) - p1_p2
        return _div(numer, denom)

    def ochiai_coeff(self):
        """Ochiai similarity coefficient (Fowlkes-Mallows)

        One interpretation of this coefficient that it is equal to the
        geometric mean of the conditional probability of an element (in the
        case of pairwise clustering comparison, a pair of elements) belonging
        to the same cluster given that they belong to the same class [1]_.

        This coefficient is in the L-family, and thus it can be corrected for
        chance by subtracting its value under fixed-margin null model. The
        resulting adjusted index is very close to, but not the same as,
        Matthews Correlation Coefficient. Empirically, the discriminating power
        of the adjusted coefficient is equal to that of Matthews' Correlation
        Coefficient to within rounding error.

        Synonyms: Cosine Similarity, Fowlkes-Mallows Index

        See Also
        --------
        jaccard_coeff, dice_coeff, ochiai_coeff_adj

        References
        ----------

        .. [1] `Ramirez, E. H., Brena, R., Magatti, D., & Stella, F. (2012).
               Topic model validation. Neurocomputing, 76(1), 125-133.
               <http://dx.doi.org/10.1016/j.neucom.2011.04.032>`_
        """
        a, c, _, b = self.to_ccw()
        p1, p2 = a + b, a + c
        if a == b == c == 0:
            return np.nan
        elif a == 0:
            return 0.0
        return _div(a, sqrt(p1 * p2))

    def sokal_sneath_coeff(self):
        """Sokal and Sneath similarity index

        In a 2x2 matrix

        .. math::

            \\begin{matrix} a & b \\\\ c & d \\end{matrix}

        Dice places more weight on :math:`a` component, Jaccard places equal
        weight on :math:`a` and :math:`b + c`, while Sokal and Sneath places
        more weight on :math:`b + c`.

        See Also
        --------
        dice_coeff, jaccard_coeff
        """
        a, c, _, b = self.to_ccw()
        return _div(a, a + 2 * (b + c))

    def prevalence_index(self):
        """Prevalence

        In interrater agreement studies, prevalence is high when the proportion
        of agreements on the positive classification differs from that of the
        negative classification.  Example of a confusion matrix with high
        prevalence of negative response (note that this happens regardless of
        which rater we look at):

        .. math::

            \\begin{matrix} 3 & 27 \\\\ 28 & 132 \\end{matrix}

        See Also
        --------

        bias_index
        """
        return _div(abs(self.TP - self.TN), self.grand_total)

    def frequency_bias(self):
        """Frequency bias

        How much more often is rater B is predicting TP
        """
        return _div(self.TP + self.FP, self.TP + self.FN)

    def bias_index(self):
        """Bias Index

        In interrater agreement studies, bias is the extent to which the raters
        disagree on the positive-negative ratio of the binary variable studied.
        Example of a confusion matrix with high bias of rater A (represented by
        rows) towards negative rating:

        .. math::

            \\begin{matrix} 17 & 14 \\\\ 78 & 81 \\end{matrix}

        See Also
        --------

        prevalence_index
        """
        return _div(abs(self.FN - self.FP), self.grand_total)

    def informedness(self):
        """Informedness (recall corrected for chance)

        A complement to markedness. Can be thought of as recall corrected for
        chance. Alternative formulations:

        .. math::

            Informedness &= Sensitivity + Specificity - 1.0 \\\\
                         &= TPR - FPR

        In the case of ranked predictions, TPR can be plotted on the y-axis
        with FPR on the x-axis. The resulting plot is known as Receiver
        Operating Characteristic (ROC) curve [1]_. The delta between a point on
        the ROC curve and the diagonal is equal to the value of informedness at
        the given FPR threshold.

        This measure was first proposed for evaluating medical diagnostics
        tests in [2]_, and was also used in meteorology under the name "True
        Skill Score" [3]_.

        Synonyms: Youden's J, True Skill Score, Hannssen-Kuiper Score,
        Attributable Risk, DeltaP.

        See Also
        --------

        markedness

        References
        ----------

        .. [1] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern
               recognition letters, 27(8), 861-874.
               <http://doi.org/10.1016/j.patrec.2005.10.010>`_

        .. [2] `Youden, W. J. (1950). Index for rating diagnostic tests. Cancer,
               3(1), 32-35.
               <http://www.ncbi.nlm.nih.gov/pubmed/15405679>`_

        .. [3] `Doswell III, C. A., Davies-Jones, R., & Keller, D. L. (1990). On
               summary measures of skill in rare event forecasting based on
               contingency tables. Weather and Forecasting, 5(4), 576-585.
               <http://journals.ametsoc.org/doi/abs/10.1175/1520-0434%281990%29005%3C0576%3AOSMOSI%3E2.0.CO%3B2>`_
        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        n = p1 + q1

        if n == 0:
            return np.nan
        elif p1 == n:
            return 0.0
            # return _div(a - b, 2 * (a + b))
        elif q1 == n:
            return 0.0
            # return _div(d - c, 2 * (d + c))
        else:
            return _div(self.covar(), p1 * q1)

    def markedness(self):
        """Markedness (precision corrected for chance)

        A complement to informedness. Can be thought of as precision corrected
        for chance. Alternative formulations:

        .. math::

            Markedness &= PPV + NPV - 1.0 \\\\
                       &= PPV - FOR

        In the case of ranked predictions, PPV can be plotted on the y-axis
        with FOR on the x-axis. The resulting plot is known as Relative
        Operating Level (ROL) curve [1]_. The delta between a point on the ROL
        curve and the diagonal is equal to the value of markedness at the given
        FOR threshold.

        Synonyms: DeltaP′

        See Also
        --------

        informedness

        References
        ----------

        .. [1] `Mason, S. J., & Graham, N. E. (2002). Areas beneath the
               relative operating characteristics (ROC) and relative
               operating levels (ROL) curves: Statistical significance
               and interpretation. Quarterly Journal of the Royal
               Meteorological Society, 128(584), 2145-2166.
               <https://doi.org/10.1256/003590002320603584>`_
        """
        a, c, d, b = self.to_ccw()
        p2, q2 = a + c, b + d
        n = p2 + q2

        if n == 0:
            return np.nan
        elif p2 == n:
            return 0.0
            # return _div(a - c, 2 * (a + c))
        elif q2 == n:
            return 0.0
            # return _div(d - b, 2 * (d + b))
        else:
            return _div(self.covar(), p2 * q2)

    def xcoeff(self):
        """Alternative to ``loevinger_coeff`` but with -1 lower bound
        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = p1 + q1

        cov = self.covar()

        if n == 0:
            return np.nan
        elif a == n or d == n:
            return 0.5
        elif b == n or c == n:
            return -1.0
        elif cov > 0.0:
            return _div(cov, min(p1 * q2, p2 * q1))
        elif cov < 0.0:
            return _div(cov, min(n * c, n * b))
        else:
            return 0.0

    def pairwise_hcv(self):
        """Pairwise homogeneity, completeness, and their geometric mean

        Each of the two one-sided measures is defined as follows:

        .. math::

            \\hat{M}_{adj} = \\frac{M - E[M]}{M_{max} - min(E[M], M)}.

        It is clear from the definition above that *iff* :math:`M < E[M]` and
        :math:`M \\leq M_{max}`, the denominator will switch from the standard
        normalization interval to a larger one, thereby ensuring that
        :math:`-1.0 \\leq \\hat{M}_{adj} \\leq 1.0`.  The definition for the
        bottom half of the range can also be expressed in terms of the standard
        adjusted value:

        .. math::

            \\hat{M}_{adj} = \\frac{M_{adj}}{(1 + |M_{adj}|^n)^{1/n}}, \\quad M_{adj} < 0, n = 1.

        The resulting measure is not symmetric over its range (negative values
        are scaled differently from positive values), however this should not
        matter for applications where negative correlation does not carry any
        special meaning other than being additional evidence for absence of
        positive correlation.  Such as a situation occurs in pairwise confusion
        matrices used in cluster analysis. Nevertheless, if more symmetric
        behavior near zero is desired, the upper part of the negative range can
        be linearized either by increasing :math:`n` in the definition above or
        by replacing it with :math:`\\hat{M}_{adj} = tanh(M_{adj})` transform.

        For the compound measure, the geometric mean was chosen over the
        harmonic after the results of a Monte Carlo power analysis, due to
        slightly better discriminating performance. For positive matrices, the
        geometric mean is equal to ``matthews_corr``, while the harmonic mean
        would have been equal to ``kappa``. For negative matrices, the harmonic
        mean would have remained monotonic (though not equal) to Kappa, while
        the geometric mean is neither monotonic nor equal to MCC, despite the
        two being closely correlated. The discriminating performance indices of
        the geometric mean and of MCC are empirically the same (equal to within
        rounding error).

        For matrices with negative covariance, it is possible to switch to
        ``markedness`` and ``informedness`` as one-sided components
        (homogeneity and completeness, respectively). However, the desirable
        property of measure orthogonality will not be preserved then, since
        markedness and informedness exhibit strong correlation under the
        assumed null model.
        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = a + b + c + d

        cov = self.covar()

        if n == 0.0:
            k0, k1, k2 = np.nan, np.nan, np.nan
        elif a == n or d == n:
            k0, k1, k2 = 0.5, 0.5, 0.5
        elif b == n:
            k0, k1, k2 = -1.0, -0.0, -0.0
        elif c == n:
            k0, k1, k2 = -0.0, -1.0, -0.0
        elif p1 == n or q2 == n:
            k0, k1, k2 = 0.0, 0.0, 0.0
        elif p2 == n or q1 == n:
            k0, k1, k2 = 0.0, 0.0, 0.0
        elif cov > 0.0:
            k0 = _div(cov, p2 * q1)
            k1 = _div(cov, p1 * q2)
            k2 = _div(cov, sqrt(p1 * q1 * p2 * q2))
        elif cov < 0.0:
            k0 = _div(cov, n * c)
            k1 = _div(cov, n * b)
            k2 = _div(cov, n * sqrt(b * c))
        else:
            k0, k1, k2 = 0.0, 0.0, 0.0

        return k0, k1, k2

    def kappas(self):
        """Pairwise precision and recall corrected for chance

        Kappa decomposes into a pair of components (regression coefficients),
        :math:`\\kappa_0` (precision-like) and :math:`\\kappa_1` (recall-like),
        of which it is a harmonic mean:

        .. math::

            \\kappa_0 = \\frac{cov}{p_2 q_1}, \\quad \\kappa_1 = \\frac{cov}{p_1 q_2}.

        These coefficients are interesting because they represent precision and
        recall, respectively, corrected for chance by subtracting the
        fixed-margin null model. In clustering context, :math:`\\kappa_0`
        corresponds to pairwise homogeneity, while :math:`\\kappa_1`
        corresponds to pairwise completeness. The geometric mean of the two
        components is equal to Matthews' Correlation Coefficient, while their
        maximum is equal to Loevinger's H when :math:`ad \\geq bc`.

        See Also
        --------

        kappa, loevinger_coeff, matthews_corr
        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = a + b + c + d

        if a == n or d == n:
            k0, k1 = np.nan, np.nan
        elif b == n:
            k0, k1 = np.NINF, -0.0
        elif c == n:
            k0, k1 = -0.0, np.NINF
        elif p1 == n or q2 == n:
            k0, k1 = np.nan, 0.0
        elif p2 == n or q1 == n:
            k0, k1 = 0.0, np.nan
        else:
            cov = self.covar()
            p2_q1, p1_q2 = p2 * q1, p1 * q2
            k0, k1 = _div(cov, p2_q1), _div(cov, p1_q2)

        return k0, k1, self.kappa()

    def loevinger_coeff(self):
        """Loevinger's Index of Homogeneity (Loevinger's H)

        Given a clustering (numbers correspond to class labels, inner groups to
        clusters) with perfect homogeneity but imperfect completeness, Loevinger
        coefficient returns a perfect score on the corresponding pairwise
        co-association matrix::

            >>> clusters = [[0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
            >>> t = ClusteringMetrics.from_clusters(clusters)
            >>> t.pairwise.loevinger_coeff()
            1.0

        At the same time, kappa and Matthews coefficients are 0.63 and 0.68,
        respectively. Loevinger coefficient will also return a perfect score
        for the dual situation::

            >>> clusters = [[0, 2, 2, 0, 0, 0], [1, 1, 1, 1]]
            >>> t = ClusteringMetrics.from_clusters(clusters)
            >>> t.pairwise.loevinger_coeff()
            1.0

        Loevinger's coefficient has a unique property: all two-way correlation
        coefficients on a 2x2 table that are in L-family (including Kappa and
        Matthews' correlation coefficient) become Loevinger's coefficient after
        normalization by maximum value [1]_. The common Precision measure
        becomes Loevinger coefficient after adjusting for 'random choice'
        precision (total frequency of positives in the population). However,
        this measure is not symmetric: when :math:`ad < bc`, it does not have a
        lower bound. For an equivalent symmetric measure, use Cole coefficient.

        See Also
        --------
        cole_coeff

        References
        ----------

        .. [1] `Warrens, M. J. (2008). On association coefficients for 2x2
               tables and properties that do not depend on the marginal
               distributions.  Psychometrika, 73(4), 777-789.
               <https://doi.org/10.1007/s11336-008-9070-3>`_

        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = p1 + q1

        cov = self.covar()

        if n == 0:
            return np.nan
        elif a == n or d == n:
            # only one (diagonal) cell is non-zero
            return 0.5
        elif cov == 0.0:
            return 0.0
        else:
            return _div(cov, min(p1 * q2, p2 * q1))

    def kappa(self):
        """Cohen's Kappa (Interrater Agreement)

        Kappa coefficient is best known in the psychology field where it was
        introduced to measure interrater agreement [1]_. It has also been used
        in replication studies [2]_, clustering evaluation [3]_, image
        segmentation [4]_, feature selection [5]_ [6]_, forecasting [7]_, and
        network link prediction [8]_. The first derivation of this measure is
        in [9]_.

        Kappa can be derived by correcting either Accuracy (Simple Matching
        Coefficient, Rand Index) or F1-score (Dice Coefficient) for chance.
        Conversely, Dice coefficient can be derived from Kappa by obtaining its
        limit as :math:`d \\rightarrow \\infty`. Normalizing Kappa by its
        maximum value given fixed-margin table gives Loevinger's H.

        Synonyms: Adjusted Rand Index, Heidke Skill Score

        See Also
        --------
        kappas, loevinger_coeff, matthews_corr, dice_coeff

        References
        ----------

        .. [1] `Cohen, J. (1960). A coefficient of agreement for nominal scales.
               Educational and psychological measurement, 20(1), 37-46.
               <https://doi.org/10.1177/001316446002000104>`_

        .. [2] `Arabie, P., Hubert, L. J., & De Soete, G. (1996). Clustering
               validation: results and implications for applied analyses (p.
               341).  World Scientific Pub Co Inc.
               <https://doi.org/10.1142/9789812832153_0010>`_

        .. [3] `Warrens, M. J. (2008). On the equivalence of Cohen's kappa and
               the Hubert-Arabie adjusted Rand index. Journal of Classification,
               25(2), 177-183.
               <https://doi.org/10.1007/s00357-008-9023-7>`_

        .. [4] `Briggman, K., Denk, W., Seung, S., Helmstaedter, M. N., &
               Turaga, S. C. (2009). Maximin affinity learning of image
               segmentation. In Advances in Neural Information Processing
               Systems (pp. 1865-1873).
               <http://books.nips.cc/papers/files/nips22/NIPS2009_0084.pdf>`_

        .. [5] `Santos, J. M., & Embrechts, M. (2009). On the use of the
               adjusted rand index as a metric for evaluating supervised
               classification. In Artificial neural networks - ICANN 2009 (pp.
               175-184).  Springer Berlin Heidelberg.
               <https://doi.org/10.1007/978-3-642-04277-5_18>`_

        .. [6] `Santos, J. M., & Ramos, S. (2010, November). Using a clustering
               similarity measure for feature selection in high dimensional
               data sets.  In Intelligent Systems Design and Applications
               (ISDA), 2010 10th International Conference on (pp. 900-905).
               IEEE.
               <http://dx.doi.org/10.1109/ISDA.2010.5687073>`_

        .. [7] `Doswell III, C. A., Davies-Jones, R., & Keller, D. L. (1990). On
               summary measures of skill in rare event forecasting based on
               contingency tables. Weather and Forecasting, 5(4), 576-585.
               <http://journals.ametsoc.org/doi/abs/10.1175/1520-0434%281990%29005%3C0576%3AOSMOSI%3E2.0.CO%3B2>`_

        .. [8] `Hoffman, M., Steinley, D., & Brusco, M. J. (2015). A note on
               using the adjusted Rand index for link prediction in networks.
               Social Networks, 42, 72-79.
               <http://dx.doi.org/10.1016/j.socnet.2015.03.002>`_

        .. [9] `Heidke, Paul. "Berechnung des Erfolges und der Gute der
               Windstarkevorhersagen im Sturmwarnungsdienst." Geografiska
               Annaler (1926): 301-349.
               <http://www.jstor.org/stable/519729>`_

        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = p1 + q1

        if n == 0:
            return np.nan
        elif a == n or d == n:
            # only one (diagonal) cell is non-zero
            return 0.5

        return _div(2 * self.covar(), p1 * q2 + p2 * q1)

    def mp_corr(self):
        """Maxwell & Pilliner's association index

        Another covariance-based association index corrected for chance. Like
        MCC, based on a mean of informedness and markedness, except uses a
        harmonic mean instead of geometric. Like Kappa, turns into Dice
        coefficient (F-score) as 'd' approaches infinity.

        On typical problems, the resolving power of this coefficient is nearly
        identical to that of Cohen's Kappa and is only very slightly below that
        of Matthews' correlation coefficient.

        See Also
        --------
        kappa, matthews_corr
        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = p1 + q1

        if n == 0:
            return np.nan
        elif a == n or d == n:
            # only one (diagonal) cell is non-zero
            return 0.5
        elif b == n or c == n:
            # only one (non-diagonal) cell is non-zero
            return -0.5

        return _div(2 * self.covar(), p1 * q1 + p2 * q2)

    def matthews_corr(self):
        """Matthews Correlation Coefficient (Phi coefficient)

        MCC is directly related to the Chi-square statistic. Its value is equal
        to the Chi-square value normalized by the maximum value the Chi-square
        can achieve with given margins (for a 2x2 table, the maximum Chi-square
        score is equal to the grand total N) transformed to correlation space
        by taking a square root.

        MCC is a also a geometric mean of informedness and markedness (the
        regression coefficients of the problem and its dual). As :math:`d
        \\rightarrow \\infty`, MCC turns into Ochiai coefficient. Unlike with
        Kappa, normalizing the corresponding similarity coefficient for chance
        by subtracting the fixed-margin null model does not produce MCC in
        return, but gives a different index with equivalent discriminating
        power to that of MCC. Normalizing MCC by its maximum value under fixed-
        margin model gives Loevinger's H.

        Empirically, the discriminating power of MCC is sligtly better than
        that of ``mp_corr`` and ``kappa``, and is only lower than that of
        ``loevinger_coeff`` under highly biased conditions. While MCC is a
        commonly used and recently preferred measure of prediction and
        reproducibility [1]_, it is somewhat strange that one can hardly find
        any literature that uses this index in clustering comparison context,
        with some rare exceptions [2]_ [3]_.

        Synonyms: Phi Coefficient, Product-Moment Correlation

        See Also
        --------
        kappa, mp_corr, ochiai_coeff

        References
        ----------

        .. [1] `MAQC Consortium. (2010). The MicroArray Quality Control
               (MAQC)-II study of common practices for the development and
               validation of microarray-based predictive models. Nature
               biotechnology, 28(8), 827-838.
               <http://doi.org/10.1038/nbt.1665>`_

        .. [2] `Xiao, J., Wang, X. F., Yang, Z. F., & Xu, C. W. (2008).
               Comparison of Supervised Clustering Methods for the Analysis of
               DNA Microarray Expression Data. Agricultural Sciences in China,
               7(2), 129-139.
               <http://dx.doi.org/10.1016/S1671-2927%2808%2960032-2>`_

        .. [3] `Kao, D. (2012). Using Matthews correlation coefficient to
               cluster annotations.  NextGenetics (personal blog).
               <http://blog.nextgenetics.net/?e=47>`_
        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = p1 + q1

        if n == 0:
            return np.nan
        elif a == n or d == n:
            # only one (diagonal) cell is non-zero
            return 0.5
        elif b == n or c == n:
            # only one (non-diagonal) cell is non-zero
            return -0.5
        elif p1 == n or p2 == n or q1 == n or q2 == n:
            # one row or column is zero, another non-zero
            return 0.0

        return _div(self.covar(), sqrt(p1 * q1 * p2 * q2))

    def mic_scores(self, mean='harmonic'):
        """Mutual information-based correlation

        The coefficient decomposes into regression coefficients defined
        according to fixed-margin tables. The ``mic1`` coefficient, for
        example, is obtained by dividing the G-score by the maximum achievable
        value on a table with fixed true class counts (which here correspond to
        row totals).  The ``mic0`` is its dual, defined by dividing the G-score
        by its maximum achievable value with fixed predicted label counts (here
        represented as column totals).

        ``mic0`` roughly corresponds to precision (homogeneity) while ``mic1``
        roughly corresponds to recall (completeness).
        """
        h, c, rsquare = self.entropy_scores(mean=mean)
        covsign = copysign(1, self.covar())
        mic0 = covsign * sqrt(c)
        mic1 = covsign * sqrt(h)
        mic2 = covsign * sqrt(rsquare)
        return mic0, mic1, mic2

    def yule_q(self):
        """Yule's Q (association index)

        Yule's Q relates to the odds ratio (DOR) as follows:

        .. math::

            Q = \\frac{DOR - 1}{DOR + 1}.

        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = a + b + c + d

        if n == 0:
            return np.nan
        elif p1 == n:
            # c and d are zero
            return _div(a - b, p1)
        elif p2 == n:
            # b and d are zero
            return _div(a - c, p2)
        elif q1 == n:
            # a and b are zero
            return _div(d - c, q1)
        elif q2 == n:
            # a and c are zero
            return _div(d - b, q2)

        return _div(self.covar(), a * d + b * c)

    def yule_y(self):
        """Yule's Y (colligation coefficient)

        The Y coefficient was used as basis of a new association
        measure by accounting for entropy in [1]_.

        References
        ----------

        .. [1] `Hasenclever, D., & Scholz, M. (2013). Comparing measures of
                association in 2x2 probability tables. arXiv preprint
                arXiv:1302.6161.
                <http://arxiv.org/pdf/1302.6161v1.pdf>`_

        """
        a, c, d, b = self.to_ccw()
        p1, q1 = a + b, c + d
        p2, q2 = a + c, b + d
        n = a + b + c + d

        if n == 0:
            return np.nan
        elif p1 == n:
            # c and d are zero
            return _div(sqrt(a) - sqrt(b), sqrt(a) + sqrt(b))
        elif p2 == n:
            # b and d are zero
            return _div(sqrt(a) - sqrt(c), sqrt(a) + sqrt(c))
        elif q1 == n:
            # a and b are zero
            return _div(sqrt(d) - sqrt(c), sqrt(d) + sqrt(c))
        elif q2 == n:
            # a and c are zero
            return _div(sqrt(d) - sqrt(b), sqrt(d) + sqrt(b))

        ad = a * d
        bc = b * c

        return _div(sqrt(ad) - sqrt(bc), sqrt(ad) + sqrt(bc))

    def cole_coeff(self):
        """Cole coefficient

        This is exactly the same coefficient as *Lewontin's D'*. It is defined
        as:

        .. math::

            D' = \\frac{cov}{cov_{max}},

        where :math:`cov_{max}` is the maximum covariance attainable under the
        given marginal distribution. When :math:`ad \\geq bc`, this coefficient
        is equivalent to Loevinger's H.

        Synonyms: C7, Lewontin's D'.

        See Also
        --------
        diseq_coeff, loevinger_coeff

        """
        return self.diseq_coeff(standardize=True)

    def diseq_coeff(self, standardize=False):
        """Linkage disequilibrium

        .. math::

            D = \\frac{a}{n} - \\frac{p_1}{n}\\frac{p_2}{n} = \\frac{cov}{n^2}

        If ``standardize=True``, this measure is further normalized to maximum
        covariance attainable under given marginal distribution, and the
        resulting index is called *Lewontin's D'*.

        See Also
        --------
        cole_coeff

        """
        cov = self.covar()
        n = self.grand_total
        if standardize:
            a, c, d, b = self.to_ccw()
            p1, q1 = a + b, c + d
            p2, q2 = a + c, b + d
            if n == 0:
                return np.nan
            elif a == n or d == n:
                # only one (diagonal) cell is non-zero
                return 0.5
            elif b == n or c == n:
                # only one (non-diagonal) cell is non-zero
                return -0.5
            elif cov > 0.0:
                cov_max = min(p1 * q2, p2 * q1)
                return _div(cov, cov_max)
            elif cov < 0.0:
                cov_max = min(p1 * p2, q1 * q2)
                return _div(cov, cov_max)
            else:
                return 0.0
        else:
            return _div(cov, n * n)

    def covar(self):
        """Covariance (determinant of a 2x2 matrix)
        """
        a, c, d, b = self.to_ccw()
        return a * d - b * c

    # various silly terminologies follow

    # information retrieval
    precision = PPV
    recall = TPR
    accuracy = ACC
    fallout = FPR

    # clinical diagnostics
    sensitivity = TPR
    specificity = TNR
    # odds_ratio = DOR
    # youden_j = informedness

    # sales/marketing
    # hit_rate = TPR
    # miss_rate = FNR

    # ecology
    # sm_coeff = ACC
    # phi_coeff = matthews_corr

    # meteorology
    # heidke_skill = kappa
    # true_skill = informedness


def mutual_info_score(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Sklean function
    """
    ct = ContingencyTable.from_labels(labels_true, labels_pred)
    return ct.mutual_info_score()


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Memory-efficient replacement for equivalently named Scikit-Learn function
    """
    ct = ContingencyTable.from_labels(labels_true, labels_pred)
    return ct.entropy_scores()


def adjusted_rand_score(labels_true, labels_pred):
    """Rand score (accuracy) corrected for chance

    This is a memory-efficient replacement for the equivalently named
    Scikit-Learn function

    In a supplement to [1]_, the following example is given::

        >>> classes = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        >>> clusters = [1, 2, 1, 2, 2, 3, 3, 3, 3, 3]
        >>> round(adjusted_rand_score(classes, clusters), 3)
        0.313

    References
    ----------

    .. [1] `Yeung, K. Y., & Ruzzo, W. L. (2001). Details of the adjusted Rand
            index and clustering algorithms, supplement to the paper "An empirical
            study on principal component analysis for clustering gene expression
            data". Bioinformatics, 17(9), 763-774.
            <http://faculty.washington.edu/kayee/pca/>`_

    """
    ct = ClusteringMetrics.from_labels(labels_true, labels_pred)
    return ct.adjusted_rand_index()


def adjusted_mutual_info_score(labels_true, labels_pred):
    """Adjusted Mutual Information for two partitions

    This is a memory-efficient replacement for the equivalently named
    Scikit-Learn function.

    Perfect labelings are both homogeneous and complete, hence AMI has the
    perfect score of one::

        >>> adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
        1.0
        >>> adjusted_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
        1.0

    If classes members are completely split across different clusters, the
    assignment is utterly incomplete, hence AMI equals zero::

        >>> adjusted_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
        0.0

    """
    ct = ContingencyTable.from_labels(labels_true, labels_pred)
    return ct.adjusted_mutual_info()


def product_moment(*args, **kwargs):
    """Return MCC score for a 2x2 contingency table
    """
    return ConfusionMatrix2.from_ccw(*args, **kwargs).matthews_corr()


def cohen_kappa(*args, **kwargs):
    """Return Cohen's Kappa for a 2x2 contingency table
    """
    return ConfusionMatrix2.from_ccw(*args, **kwargs).kappa()
