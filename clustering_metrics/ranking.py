"""

Motivation
----------

Assume that there is a data set of mostly unique samples where a hidden binary
variable is dependent on the number of similar samples that exist in the set
(i.e. a sample is called positive if it has many neighbors) and that our goal
is to label all samples in this set. Given sparse enough data, if a clustering
method relies on the same sample property on which the ground truth similarity
space is defined, it will naturally separate the samples into two groups --
those found in clusters and containing mostly positives, and those found
outside clusters and containing mostly negatives.  There would exist only one
possible perfect clustering---one with a single, entirely homogeneous cluster C
that covers all positives present in the data set. If we were to produce such a
clustering, we could correctly label all positive samples in one step with the
simple rule, *all positive samples belong to cluster C*. Under an imperfect
clustering, however, the presence of the given sample in a cluster of size two
or more implies the sample is only somewhat more likely to be positive, with
the confidence of the positive call monotonously increasing with the size of
the cluster.  In other words, our expectation from a good clustering is that it
will help us minimize the amount of work labeling samples.

This idea for this metric originated when mining for positive spam examples in
large data sets of short user-generated content.  Given large enough data sets,
spam content naturally forms clusters either because creative rewriting of
every single individual spam message is too expensive for spammers to employ,
or because, even if human or algorithmic rewriting is applied, one can still
find features that link individual spam messages to their creator or to the
product or service being promoted in the spam campaign. The finding was
consistent with what is reported in literature [104]_.

Algorithm
---------

Given a clustering, we order the clusters from the largest one to the smallest
one. We then plot a cumulative step function where the width of the bin under a
given "step" is proportional to cluster size, and the height of the bin is
proportional to the expected number of positive samples seen so far [103]_. If a
sample is in a cluster of size one, we assume it is likely to be negative and
is therefore checked on an individual basis (the specific setting of cluster
size at which the expectation changes is our 'threshold' parameter. The result
of this assumption is that the expected contribution from unclustered
samples is equal to their actual contribution (we assume individual checking
always gives a correct answer). After two-way normalization, a perfect
clustering (i.e. where a single perfectly homogeneous cluster covers the entire
set of positives) will have the AUL score of 1.0. A failure to will result in
the AUL of 0.5. A perverse clustering, i.e. one where many negative samples fall
into clusters whose size is above our threshold, or where many positive samples
remain unclustered (fall into clusters of size below the threshold one) the AUL
somewhere between 0.0 and 0.5.

A special treatment is necessary for cases where clusters are tied by size. If
one were to treat tied clusters as a single group, one would obtain AUL of 1.0
when no clusters at all are present, which is against our desiderata.  On the
other hand, if one were to treat tied clusters entirely separately, one would
obtain different results depending on the properties of the sorting algorithm,
also an undesirable situation. Always placing "heavy" clusters (i.e. those
containing more positives) towards the beginning or towards the end of the tied
group will result in, respectively, overestimating or underestimating the true
AUL. The solution here is to average the positive counts among all clusters in a
tied group, and then walk through them one by one, with the stepwise cumulative
function asymptotically approaching a diagonal from the group's bottom left
corner to the top right one. This way, a complete absence of clustering (i.e.
all clusters are of size one) will always result in AUL of 0.5.

The resulting AUL measure has some similarity with the Gini coefficient of
inequality [105]_ except we plot the corresponding curve in the opposite
direction (from "richest" to "poorest"), and do not subtract 0.5 from the
resulting score.


.. [103] We take the expected number of positives and not the actual number seen
       so far as the vertical scale in order to penalize non-homogeneous
       clusters. Otherwise the y=1.0 ceiling would be reached early in the
       process even in very bad cases, for example when there is only one giant
       non-homogeneous cluster.

References
----------

.. [104] `Whissell, J. S., & Clarke, C. L. (2011, September). Clustering for
         semi-supervised spam filtering. In Proceedings of the 8th Annual
         Collaboration, Electronic messaging, Anti-Abuse and Spam Conference
         (pp. 125-134). ACM.
         <https://doi.org/10.1145/2030376.2030391>`_

.. [105] `Wikipedia entry for Gini coefficient of inequality
         <https://en.wikipedia.org/wiki/Gini_coefficient>`_

"""

import warnings
import numpy as np
from itertools import izip, chain
from operator import itemgetter
from pymaptools.iter import aggregate_tuples
from pymaptools.containers import labels_to_clusters
from clustering_metrics.skutils import auc, roc_curve


def num2bool(num):
    """True if zero or positive real, False otherwise

    When binarizing class labels, this lets us be consistent with Scikit-Learn
    where binary labels can be {0, 1} with 0 being negative or {-1, 1} with -1
    being negative.

    """
    return num > 0


class LiftCurve(object):

    """Lift Curve for cluster-size correlated classification

    """

    def __init__(self, score_groups):
        self._score_groups = list(score_groups)

    @classmethod
    def from_counts(cls, counts_true, counts_pred):
        """Instantiates class from arrays of true and predicted counts

        Parameters
        ----------

        counts_true : array, shape = [n_clusters]
            Count of positives in cluster

        counts_pred : array, shape = [n_clusters]
            Predicted number of positives in each cluster

        """

        # convert input to a series of tuples
        count_groups = izip(counts_pred, counts_true)

        # sort tuples by predicted count in descending order
        count_groups = sorted(count_groups, key=itemgetter(0), reverse=True)

        # group tuples by predicted count so as to handle ties correctly
        return cls(aggregate_tuples(count_groups))

    @classmethod
    def from_clusters(cls, clusters, is_class_pos=num2bool):
        """Instantiates class from clusters of class-coded points

        Parameters
        ----------

        clusters : collections.Iterable
            List of lists of class labels

        is_class_pos: label_true -> Bool
            Boolean predicate used to binarize true (class) labels

        """
        # take all non-empty clusters, score them by size and by number of
        # ground truth positives
        data = ((len(cluster), sum(is_class_pos(class_label) for class_label in cluster))
                for cluster in clusters if cluster)
        scores_pred, scores_true = zip(*data) or ([], [])
        return cls.from_counts(scores_true, scores_pred)

    @classmethod
    def from_labels(cls, labels_true, labels_pred, is_class_pos=num2bool):
        """Instantiates class from arrays of classes and cluster sizes

        Parameters
        ----------

        labels_true : array, shape = [n_samples]
            Class labels. If binary, 'is_class_pos' is optional

        labels_pred : array, shape = [n_samples]
            Cluster labels to evaluate

        is_class_pos: label_true -> Bool
            Boolean predicate used to binarize true (class) labels

        """
        clusters = labels_to_clusters(labels_true, labels_pred)
        return cls.from_clusters(clusters, is_class_pos=is_class_pos)

    def aul_score(self, threshold=1, plot=False):
        """Calculate AUL score

        Parameters
        ----------

        threshold : int, optional (default=1)
            only predicted scores above this number considered accurate

        plot : bool, optional (default=False)
            whether to return X and Y data series for plotting

        """
        total_any = 0
        total_true = 0
        assumed_vertical = 0
        aul = 0.0

        if plot:
            xs, ys = [], []
            bin_height = 0.0
            bin_right_edge = 0.0

        # second pass: iterate over each group of predicted scores of the same
        # size and calculate the AUL metric
        for pred_score, true_scores in self._score_groups:
            # number of clusters
            num_true_scores = len(true_scores)

            # sum total of positives in all clusters of given size
            group_height = sum(true_scores)

            total_true += group_height

            # cluster size x number of clusters of given size
            group_width = pred_score * num_true_scores

            total_any += group_width

            if pred_score > threshold:
                # penalize non-homogeneous clusters simply by assuming that they
                # are homogeneous, in which case their expected vertical
                # contribution should be equal to their horizontal contribution.
                height_incr = group_width
            else:
                # clusters of size one are by definition homogeneous so their
                # expected vertical contribution equals sum total of any
                # remaining true positives.
                height_incr = group_height

            assumed_vertical += height_incr

            if plot:
                avg_true_score = group_height / float(num_true_scores)
                for _ in true_scores:
                    bin_height += avg_true_score
                    aul += bin_height * pred_score
                    if plot:
                        xs.append(bin_right_edge)
                        bin_right_edge += pred_score
                        xs.append(bin_right_edge)
                        ys.append(bin_height)
                        ys.append(bin_height)
            else:
                # if not tasked with generating plots, use a geometric method
                # instead of looping
                aul += (total_true * group_width -
                        ((num_true_scores - 1) * pred_score * group_height) / 2.0)

        if total_true > total_any:
            warnings.warn(
                "Number of positives found (%d) exceeds total count of %d"
                % (total_true, total_any)
            )

        rect_area = assumed_vertical * total_any

        # special case: since normalizing the AUL defines it as always smaller
        # than the bounding rectangle, when denominator in the expression below
        # is zero, the AUL score is also equal to zero.
        aul = 0.0 if rect_area == 0 else aul / rect_area

        if plot:
            xs = np.array(xs, dtype=float) / total_any
            ys = np.array(ys, dtype=float) / assumed_vertical
            return aul, xs, ys
        else:
            return aul

    def plot(self, threshold=1, fill=True, marker=None, save_to=None):  # pragma: no cover
        """Create a graphical representation of Lift Curve

        Requires Matplotlib

        Parameters
        ----------
        threshold : int, optional (default=1)
            only predicted scores above this number considered accurate

        marker : str, optional (default=None)
            Whether to draw marker at each bend

        save_to : str, optional (default=None)
            If specified, save the plot to path instead of displaying

        """
        from matplotlib import pyplot as plt

        score, xs, ys = self.aul_score(threshold=threshold, plot=True)
        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker=marker, linestyle='-')
        if fill:
            ax.fill([0.0] + list(xs) + [1.0], [0.0] + list(ys) + [0.0], 'b', alpha=0.2)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='grey')
        ax.plot([0.0, 1.0], [1.0, 1.0], linestyle='--', color='grey')
        ax.plot([1.0, 1.0], [0.0, 1.0], linestyle='--', color='grey')
        ax.set_xlim(xmin=0.0, xmax=1.03)
        ax.set_ylim(ymin=0.0, ymax=1.04)
        ax.set_xlabel("portion total")
        ax.set_ylabel("portion expected positive")
        ax.set_title("Lift Curve (AUL=%.3f)" % score)
        if save_to is None:
            fig.show()
        else:
            fig.savefig(save_to)
            plt.close(fig)


def aul_score_from_clusters(clusters):
    """Calculate AUL score given clusters of class-coded points

    Parameters
    ----------

    clusters : collections.Iterable
         List of clusters where each point is binary-coded according to true
         class.

    Returns
    -------

    aul : float
    """
    return LiftCurve.from_clusters(clusters).aul_score()


def aul_score_from_labels(y_true, labels_pred):
    """AUL score given array of classes and array of cluster sizes

    Parameters
    ----------

    y_true : array, shape = [n_samples]
         True binary labels in range {0, 1}

    labels_pred : array, shape = [n_samples]
         Cluster labels to evaluate

    Returns
    -------

    aul : float
    """
    return LiftCurve.from_labels(y_true, labels_pred).aul_score()


class RocCurve(object):

    """Receiver Operating Characteristic (ROC)

    ::

        >>> c = RocCurve.from_labels([0, 0, 1, 1],
        ...                          [0.1, 0.4, 0.35, 0.8])
        >>> c.auc_score()
        0.75
        >>> c.max_informedness()
        0.5

    """
    def __init__(self, fprs, tprs, thresholds=None, pos_label=None,
                 sample_weight=None):
        self.fprs = fprs
        self.tprs = tprs
        self.thresholds = thresholds
        self.pos_label = pos_label
        self.sample_weight = sample_weight

    def plot(self, fill=True, marker=None, save_to=None):  # pragma: no cover
        """Plot the ROC curve
        """
        from matplotlib import pyplot as plt

        score = self.auc_score()
        xs, ys = self.fprs, self.tprs

        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker=marker, linestyle='-')
        if fill:
            ax.fill([0.0] + list(xs) + [1.0], [0.0] + list(ys) + [0.0], 'b', alpha=0.2)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle='--', color='grey')
        ax.plot([0.0, 1.0], [1.0, 1.0], linestyle='--', color='grey')
        ax.plot([1.0, 1.0], [0.0, 1.0], linestyle='--', color='grey')
        ax.set_xlim(xmin=0.0, xmax=1.03)
        ax.set_ylim(ymin=0.0, ymax=1.04)
        ax.set_ylabel('TPR')
        ax.set_xlabel('FPR')
        ax.set_title("ROC Curve (AUC=%.3f)" % score)
        if save_to is None:
            fig.show()
        else:
            fig.savefig(save_to)
            plt.close(fig)

    @classmethod
    def from_scores(cls, scores_neg, scores_pos):
        """Instantiate given scores of two ground truth classes

        The score arrays don't have to be the same length.
        """

        scores_pos = ((1, x) for x in scores_pos if not np.isnan(x))
        scores_neg = ((0, x) for x in scores_neg if not np.isnan(x))
        all_scores = zip(*chain(scores_neg, scores_pos)) or ([], [])
        return cls.from_labels(*all_scores)

    @classmethod
    def from_labels(cls, labels_true, y_score, is_class_pos=num2bool):
        """Instantiate assuming binary labeling of {0, 1}

        labels_true : array, shape = [n_samples]
            Class labels. If binary, 'is_class_pos' is optional

        y_score : array, shape = [n_samples]
            Predicted scores

        is_class_pos: label_true -> Bool
            Boolean predicate used to binarize true (class) labels
        """

        # num2bool Y labels
        y_true = map(is_class_pos, labels_true)

        # calculate axes
        fprs, tprs, thresholds = roc_curve(
            y_true, y_score, pos_label=True)

        return cls(fprs, tprs, thresholds=thresholds)

    @classmethod
    def from_clusters(cls, clusters, is_class_pos=num2bool):
        """Instantiates class from clusters of class-coded points

        Parameters
        ----------

        clusters : collections.Iterable
            List of lists of class labels

        is_class_pos: label_true -> Bool
            Boolean predicate used to binarize true (class) labels

        """
        y_true = []
        y_score = []
        for cluster in clusters:
            pred_cluster = len(cluster)
            for point in cluster:
                true_cluster = is_class_pos(point)
                y_true.append(true_cluster)
                y_score.append(pred_cluster)
        return cls.from_labels(y_true, y_score)

    def auc_score(self):
        """Replacement for Scikit-Learn's method

        If number of Y classes is other than two, a warning will be triggered
        but no exception thrown (the return value will be a NaN).  Also, we
        don't reorder arrays during ROC calculation since they are assumed to be
        in order.
        """
        return auc(self.fprs, self.tprs, reorder=False)

    def optimal_cutoff(self, scoring_method):
        """Optimal cutoff point on ROC curve under scoring method

        The scoring method must take two arguments: fpr and tpr.
        """
        max_index = np.NINF
        opt_pair = (np.nan, np.nan)
        for pair in izip(self.fprs, self.tprs):
            index = scoring_method(*pair)
            if index > max_index:
                opt_pair = pair
                max_index = index
        return opt_pair, max_index

    @staticmethod
    def _informedness(fpr, tpr):
        return tpr - fpr

    def max_informedness(self):
        """Maximum value of Informedness (TPR minus FPR) on a ROC curve

        A diagram of what this measure looks like is shown in [101]_. Note a
        correspondence between the definitions of this measure and that of
        Kolmogorov-Smirnov's supremum statistic.

        References
        ----------

        .. [101] `Wikipedia entry for Youden's J statistic
                 <https://en.wikipedia.org/wiki/Youden%27s_J_statistic>`_
        """
        return self.optimal_cutoff(self._informedness)[1]


def roc_auc_score(y_true, y_score, sample_weight=None):
    """AUC score for a ROC curve

    Replaces Scikit Learn implementation (given binary ``y_true``).
    """
    return RocCurve.from_labels(y_true, y_score).auc_score()


def dist_auc(scores0, scores1):
    """AUC score for two distributions, with NaN correction

    Note: arithmetic mean appears to be appropriate here, as other means don't
    result in total of 1.0 when sides are switched.
    """
    scores0_len = len(scores0)
    scores1_len = len(scores1)

    scores0p = [x for x in scores0 if not np.isnan(x)]
    scores1p = [x for x in scores1 if not np.isnan(x)]

    scores0n_len = scores0_len - len(scores0p)
    scores1n_len = scores1_len - len(scores1p)

    # ``nan_pairs`` are pairs for which it is impossible to define order, due
    # to at least one of the members of each being a NaN. ``def_pairs`` are
    # pairs for which order can be established.
    all_pairs = 2 * scores0_len * scores1_len
    nan_pairs = scores0n_len * scores1_len + scores1n_len * scores0_len
    def_pairs = all_pairs - nan_pairs

    # the final score is the average of the score for the defined portion and
    # of random-chance AUC (0.5), weighted according to the number of pairs in
    # each group.
    auc_score = RocCurve.from_scores(scores0p, scores1p).auc_score()
    return np.average([auc_score, 0.5], weights=[def_pairs, nan_pairs])
