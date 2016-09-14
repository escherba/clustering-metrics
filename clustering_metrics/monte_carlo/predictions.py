import numpy as np
import os
import warnings
import random
import sys
import logging
import scipy
from itertools import product, izip, chain, cycle
from collections import defaultdict
from functools import partial
from pymaptools.iter import izip_with_cycles, isiterable, take
from pymaptools.containers import labels_to_clusters, clusters_to_labels
from pymaptools.sample import discrete_sample, freqs2probas, randround
from pymaptools.io import GzipFileType, PathArgumentParser, write_json_line, read_json_lines, ndjson2col
from pymaptools.benchmark import PMTimer

from clustering_metrics.monte_carlo import utils
from clustering_metrics.fent import minmaxr
from clustering_metrics.utils import _div
from clustering_metrics.metrics import ClusteringMetrics, ConfusionMatrix2
from clustering_metrics.ranking import dist_auc
from clustering_metrics.skutils import auc


def parse_args(args=None):
    parser = PathArgumentParser()

    parser.add_argument(
        '--logging', type=str, default='WARN', help="Logging level",
        choices=[key for key in logging._levelNames.keys() if isinstance(key, str)])

    subparsers = parser.add_subparsers()

    p_mapper = subparsers.add_parser('mapper')
    p_mapper.add_argument('--h0_err', type=float, default=1.0,
                          help='H0 error rate')
    p_mapper.add_argument('--h1_err', type=float, default=0.5,
                          help='H1 error rate')
    p_mapper.add_argument('--population_size', type=int, default=2000,
                          help='population size')
    p_mapper.add_argument('--sim_size', type=int, default=1000,
                          help='Simulation size')
    p_mapper.add_argument('--nclusters', type=int, default=20,
                          help='number of clusters to generate')
    p_mapper.add_argument('--join_negatives', type=int, default=0,
                          help='whether to join negatives (if split_join<0)')
    p_mapper.add_argument('--split_join', type=int, default=0,
                          help='number of splits (if positive) or joins (if negative) to perform')
    p_mapper.add_argument('--sampling_warnings', type=int, default=0,
                          help='if true, show sampling warnings')
    p_mapper.add_argument('--output', type=GzipFileType('w'),
                          default=sys.stdout, help='Output file')
    p_mapper.add_argument('--metrics', type=str, required=True, nargs='*',
                          help='Which metrics to compute')
    p_mapper.set_defaults(func=do_mapper)

    p_reducer = subparsers.add_parser('reducer')
    p_reducer.add_argument(
        '--group_by', type=str, default=None,
        help='Field to group by')
    p_reducer.add_argument(
        '--x_axis', type=str, default=None,
        help='Which column to plot as X axis')
    p_reducer.add_argument(
        '--metrics', type=str, required=True, nargs='*',
        help='Which metrics to compute')
    p_reducer.add_argument(
        '--input', type=GzipFileType('r'), default=sys.stdin, help='File input')
    p_reducer.add_argument(
        '--output', type=str, metavar='DIR', help='Output directory')
    p_reducer.add_argument(
        '--fig_title', type=str, default=None, help='Title (for figures generated)')
    p_reducer.add_argument(
        '--fig_format', type=str, default='svg', help='Figure format')
    p_reducer.add_argument(
        '--legend_loc', type=str, default='lower left',
        help='legend location')
    p_reducer.set_defaults(func=do_reducer)

    namespace = parser.parse_args(args)
    return namespace


def do_mapper(args):
    params = dict(
        n=args.sim_size,
        nclusters=args.nclusters,
        split_join=args.split_join,
        join_negatives=bool(args.join_negatives),
        population_size=args.population_size,
        with_warnings=args.sampling_warnings,
    )
    h0 = Grid.with_sim_clusters(p_err=args.h0_err, **params)
    h1 = Grid.with_sim_clusters(p_err=args.h1_err, **params)
    with PMTimer() as timer:
        results = h0.compare(h1, args.metrics)
    for result in results:
        result.update(timer.to_dict())
        result.update(utils.serialize_args(args))
        write_json_line(args.output, result)


def auc_xscaled(xs, ys):
    """AUC score scaled to fill x interval
    """
    xmin, xmax = minmaxr(xs)
    denom = float(xmax - xmin)
    xs_corr = [(x - xmin) / denom for x in xs]
    return auc(xs_corr, ys)


def create_plots(args, df):
    import jinja2
    import matplotlib.pyplot as plt
    from palettable import colorbrewer
    from matplotlib.font_manager import FontProperties

    fontP = FontProperties()
    fontP.set_size('xx-small')

    #groups = df.set_index(args.x_axis).groupby([args.group_by])
    groups = df.groupby([args.group_by])
    metrics = list(set(args.metrics) & set(df.keys()))
    colors = take(len(metrics), cycle(chain(
        colorbrewer.qualitative.Dark2_8.mpl_colors,
        colorbrewer.qualitative.Set2_8.mpl_colors,
    )))

    template_loader = jinja2.FileSystemLoader(os.path.join(args.output, '..'))
    template_env = jinja2.Environment(loader=template_loader)
    template_interactive = template_env.get_template('template_fig_interactive.html')
    template_static = template_env.get_template('template_fig_static.html')

    table_interactive = []
    table_static = []

    for group_name, group in groups:

        # always sort by X values
        group = group.sort([args.x_axis])

        if args.fig_title is None:
            fig_title = '%s=%s' % (args.group_by, group_name)
        else:
            fig_title = args.fig_title

        # compute AUC scores
        ys = []
        for metric, color in zip(metrics, colors):
            series = group[metric]
            score = auc_xscaled(group[args.x_axis].values, series.values)
            label = "%s (%.4f)" % (metric, score)
            ys.append((score, metric, label, color))
        ys.sort(reverse=True)

        lbls_old, lbls_new, colors = zip(*ys)[1:4]
        group = group[[args.x_axis] + list(lbls_old)] \
            .set_index(args.x_axis) \
            .rename(columns=dict(zip(lbls_old, lbls_new)))

        # create plots
        fig, ax = plt.subplots()
        group.plot(ax=ax, title=fig_title, color=list(colors))
        ax.set_xlim(*minmaxr(group.index.values))
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc=args.legend_loc, prop=fontP)
        fig_name = 'fig-%s.%s' % (group_name, args.fig_format)
        fig_path = os.path.join(args.output, fig_name)
        csv_name = 'fig-%s.csv' % group_name
        csv_path = os.path.join(args.output, csv_name)
        group.to_csv(csv_path)

        table_interactive.append((
            csv_name,
            args.x_axis,
            "%s=%s" % (args.group_by, group_name),
        ))
        table_static.append(fig_name)

        fig.savefig(fig_path, format=args.fig_format)
        plt.close(fig)

    with open(os.path.join(args.output, 'fig_interactive.html'), 'w') as fh:
        fh.write(template_interactive.render(table=table_interactive))

    with open(os.path.join(args.output, 'fig_static.html'), 'w') as fh:
        fh.write(template_static.render(table=table_static))


def do_reducer(args):
    import pandas as pd
    obj = ndjson2col(read_json_lines(args.input))
    df = pd.DataFrame.from_dict(obj)
    csv_path = os.path.join(args.output, "summary.csv")
    logging.info("Writing brief summary to %s", csv_path)
    df.to_csv(csv_path)
    create_plots(args, df)


def run(args):
    logging.basicConfig(level=getattr(logging, args.logging))
    args.func(args)


def get_conf(obj):
    try:
        return obj.pairwise
    except AttributeError:
        return obj


def sample_with_error(label, error_distribution, null_distribution):
    """Return label given error probability and null distributions

    error_distribution must be of form {False: 1.0 - p_err, True: p_err}
    """
    if discrete_sample(error_distribution):
        # to generate error properly, draw from null distribution
        return discrete_sample(null_distribution)
    else:
        # no error: append actual class label
        return label


def relabel_negatives(clusters):
    """Place each negative label in its own class
    """
    idx = -1
    relabeled = []
    for cluster in clusters:
        relabeled_cluster = []
        for class_label in cluster:
            if class_label <= 0:
                class_label = idx
            relabeled_cluster.append(class_label)
            idx -= 1
        relabeled.append(relabeled_cluster)

    return relabeled


def join_clusters(clusters):
    """Reduce number of clusters 2x by joining
    """
    even = clusters[0::2]
    odd = clusters[1::2]
    if len(even) < len(odd):
        even.append([])
    elif len(even) > len(odd):
        odd.append([])
    assert len(even) == len(odd)

    result = []
    for c1, c2 in izip(even, odd):
        result.append(c1 + c2)
    return result


def split_clusters(clusters):
    """Increase number of clusters 2x by splitting
    """
    result = []
    for cluster in clusters:
        even = cluster[0::2]
        odd = cluster[1::2]
        assert len(even) + len(odd) == len(cluster)
        if even:
            result.append(even)
        if odd:
            result.append(odd)
    return result


def simulate_clustering(galpha=2, gbeta=10, nclusters=20, pos_ratio=0.2,
                        p_err=0.05, population_size=2000, split_join=0,
                        join_negatives=False, with_warnings=True):

    if not 0.0 <= p_err <= 1.0:
        raise ValueError(p_err)

    csizes = map(randround, np.random.gamma(galpha, gbeta, nclusters))

    # make sure at least one cluster is generated
    num_pos = sum(csizes)
    if num_pos == 0:
        csizes.append(1)
        num_pos += 1

    num_neg = max(0, population_size - num_pos)

    if with_warnings:
        if not 0.0 <= pos_ratio <= 1.0:
            raise ValueError(pos_ratio)
        expected_num_neg = num_pos * _div(1.0 - pos_ratio, pos_ratio)
        actual_neg_ratio = _div(num_neg - expected_num_neg, expected_num_neg)
        if abs(actual_neg_ratio) > 0.2:
            warnings.warn(
                "{:.1%} {} negatives than expected. Got: {} "
                "(expected: {}. Recommended population_size: {})"
                .format(abs(actual_neg_ratio), ("fewer" if actual_neg_ratio < 0.0 else "more"), num_neg,
                        int(expected_num_neg), int(expected_num_neg + num_pos)))

    # set up probability distributions we will use
    null_dist = freqs2probas([num_neg] + csizes)
    error_dist = {False: 1.0 - p_err, True: p_err}

    # negative case first
    negatives = []
    for _ in xrange(num_neg):
        class_label = sample_with_error(0, error_dist, null_dist)
        negatives.append([class_label])

    # positive cases
    positives = []
    for idx, csize in enumerate(csizes, start=1):
        if csize < 1:
            continue
        cluster = []
        for _ in xrange(csize):
            class_label = sample_with_error(idx, error_dist, null_dist)
            cluster.append(class_label)
        positives.append(cluster)

    if split_join > 0:
        for _ in xrange(split_join):
            positives = split_clusters(positives)
    elif split_join < 0:
        for _ in xrange(-split_join):
            positives = join_clusters(positives)
        if join_negatives:
            for _ in xrange(-split_join):
                negatives = join_clusters(negatives)

    return relabel_negatives(positives + negatives)


def simulate_labeling(sample_size=2000, **kwargs):

    clusters = simulate_clustering(**kwargs)
    tuples = zip(*clusters_to_labels(clusters))
    random.shuffle(tuples)
    tuples = tuples[:sample_size]
    ltrue, lpred = zip(*tuples) or ([], [])
    return ltrue, lpred


class Grid(object):

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.max_classes = None
        self.max_counts = None
        self.n = None
        self.size = None

        self.grid = None
        self.grid_type = None
        self.get_matrix = None
        self.show_record = None

    @classmethod
    def with_sim_clusters(cls, n=1000, size=200, seed=None, **kwargs):
        obj = cls(seed=seed)

        obj.grid = obj.fill_sim_clusters(size=size, n=n, **kwargs)
        obj.grid_type = 'sim_clusters'
        obj.get_matrix = obj.matrix_from_labels
        obj.show_record = obj.show_cluster
        return obj

    @classmethod
    def with_clusters(cls, n=1000, size=200, max_classes=5, seed=None):
        obj = cls(seed=seed)

        obj.grid = obj.fill_clusters(max_classes=max_classes, size=size, n=n)
        obj.grid_type = 'clusters'
        obj.get_matrix = obj.matrix_from_labels
        obj.show_record = obj.show_cluster
        return obj

    @classmethod
    def with_matrices(cls, n=1000, max_counts=100, seed=None):
        obj = cls(seed=seed)

        obj.grid = obj.fill_matrices(max_counts=max_counts, n=n)
        obj.grid_type = 'matrices'
        obj.get_matrix = obj.matrix_from_matrices
        obj.show_record = obj.show_matrix
        return obj

    def show_matrix(self, idx, inverse=False):
        grid = self.grid
        return grid[0][idx]

    def show_cluster(self, idx, inverse=False):
        grid = self.grid
        a, b = (1, 0) if inverse else (0, 1)
        return labels_to_clusters(grid[a][idx], grid[b][idx])

    def best_clustering_by_score(self, score, flip_sign=False):
        idx, val = self.find_highest(score, flip_sign)
        return {"idx": idx,
                "found": "%s = %.4f" % (score, val),
                "result": self.show_cluster(idx),
                "inverse": self.show_cluster(idx, inverse=True)}

    @staticmethod
    def matrix_from_labels(*args):
        ltrue, lpred = args
        return ClusteringMetrics.from_labels(ltrue, lpred)

    @staticmethod
    def matrix_from_matrices(*args):
        arr = args[0]
        return ConfusionMatrix2.from_ccw(*arr)

    def iter_grid(self):
        return enumerate(izip(*self.grid))

    iter_clusters = iter_grid

    def iter_matrices(self):
        if self.grid_type in ['matrices']:
            for idx, tup in self.iter_grid():
                yield idx, self.matrix_from_matrices(*tup)
        elif self.grid_type in ['clusters', 'sim_clusters']:
            for idx, labels in self.iter_grid():
                yield idx, self.matrix_from_labels(*labels)

    def describe_matrices(self):
        for idx, matrix in self.iter_matrices():
            tup = tuple(get_conf(matrix).to_ccw())
            max_idx = tup.index(max(tup))
            if max_idx != 2:
                print idx, tup

    def fill_clusters(self, n=None, size=None, max_classes=None):
        if n is None:
            n = self.n
        else:
            self.n = n
        if size is None:
            size = self.size
        else:
            self.size = size
        if max_classes is None:
            max_classes = self.max_classes
        else:
            self.max_classes = max_classes

        classes = np.random.randint(
            low=0, high=max_classes, size=(n, size))
        clusters = np.random.randint(
            low=0, high=max_classes, size=(n, size))
        return classes, clusters

    def fill_sim_clusters(self, n=None, size=None, **kwargs):
        if n is None:
            n = self.n
        else:
            self.n = n
        if size is None:
            size = self.size
        else:
            self.size = size

        classes = np.empty((n, size), dtype=np.int64)
        clusters = np.empty((n, size), dtype=np.int64)
        for idx in xrange(n):
            ltrue, lpred = simulate_labeling(sample_size=size, **kwargs)
            classes[idx, :] = ltrue
            clusters[idx, :] = lpred
        return classes, clusters

    def fill_matrices(self, max_counts=None, n=None):
        if max_counts is None:
            max_counts = self.max_counts
        else:
            self.max_counts = max_counts
        if n is None:
            n = self.n
        else:
            self.n = n

        matrices = np.random.randint(
            low=0, high=max_counts, size=(n, 4))
        return (matrices,)

    def find_highest(self, score, flip_sign=False):
        best_index = -1
        if flip_sign:
            direction = 1
            curr_score = float('inf')
        else:
            direction = -1
            curr_score = float('-inf')
        for idx, conf in self.iter_matrices():
            new_score = conf.get_score(score)
            if cmp(curr_score, new_score) == direction:
                best_index = idx
                curr_score = new_score
        return (best_index, curr_score)

    def find_matching_matrix(self, matches):
        for idx, mx in self.iter_matrices():
            mx = get_conf(mx)
            if matches(mx):
                return idx, mx

    def compute(self, scores, show_progress=False, dtype=np.float16):
        result = defaultdict(partial(np.empty, (self.n,), dtype=dtype))
        if not isiterable(scores):
            scores = [scores]
        for idx, conf in self.iter_matrices():
            if show_progress:
                pct_done = 100 * idx / float(self.n)
                if pct_done % 5 == 0:
                    sys.stderr.write("%d%% done\n" % pct_done)
            for score in scores:
                score_arr = conf.get_score(score)
                if isiterable(score_arr):
                    for j, val in enumerate(score_arr):
                        result["%s-%d" % (score, j)][idx] = val
                else:
                    result[score][idx] = score_arr
        return result

    def compare(self, others, scores, dtype=np.float16, plot=False):
        result0 = self.compute(scores, dtype=dtype)

        if not isiterable(others):
            others = [others]

        result_grid = []
        for other in others:
            result1 = other.compute(scores, dtype=dtype)

            if plot:
                from matplotlib import pyplot as plt
                from palettable import colorbrewer
                colors = colorbrewer.get_map('Set1', 'qualitative', 9).mpl_colors

            result_row = {}
            for score_name, scores0 in result0.iteritems():
                scores1 = result1[score_name]
                auc_score = dist_auc(scores0, scores1)
                result_row[score_name] = auc_score
                if plot:
                    scores0p = [x for x in scores0 if not np.isnan(x)]
                    scores1p = [x for x in scores1 if not np.isnan(x)]
                    hmin0, hmax0 = minmaxr(scores0p)
                    hmin1, hmax1 = minmaxr(scores1p)
                    bins = np.linspace(min(hmin0, hmin1), max(hmax0, hmax1), 50)
                    plt.hist(scores0p, bins, alpha=0.5, label='0', color=colors[0], edgecolor="none")
                    plt.hist(scores1p, bins, alpha=0.5, label='1', color=colors[1], edgecolor="none")
                    plt.legend(loc='upper right')
                    plt.title("%s: AUC=%.4f" % (score_name, auc_score))
                    plt.show()
            result_grid.append(result_row)
        return result_grid

    def corrplot(self, compute_result, save_to, symmetric=False, **kwargs):
        items = compute_result.items()
        if not os.path.exists(save_to):
            os.mkdir(save_to)
        elif not os.path.isdir(save_to):
            raise IOError("save_to already exists and is a file")

        seen_pairs = set()
        for (lbl1, arr1), (lbl2, arr2) in product(items, items):
            if lbl1 == lbl2:
                continue
            elif (not symmetric) and (lbl2, lbl1) in seen_pairs:
                continue
            elif (not symmetric) and (lbl1, lbl2) in seen_pairs:
                continue
            figtitle = "%s vs. %s" % (lbl1, lbl2)
            filename = "%s_vs_%s.png" % (lbl1, lbl2)
            filepath = os.path.join(save_to, filename)
            if os.path.exists(filepath):
                warnings.warn("File exists: not overwriting %s" % filepath)
                seen_pairs.add((lbl1, lbl2))
                seen_pairs.add((lbl2, lbl1))
                continue

            self.plot([(arr1, arr2)], save_to=filepath, title=figtitle,
                      xlabel=lbl1, ylabel=lbl2, **kwargs)
            seen_pairs.add((lbl1, lbl2))
            seen_pairs.add((lbl2, lbl1))

    @staticmethod
    def plot(pairs, xlim=None, ylim=None, title=None, dither=0.0002,
             marker='.', s=0.01, color='black', alpha=1.0, save_to=None,
             label=None, xlabel=None, ylabel=None, **kwargs):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for (xs, ys), dither_, marker_, s_, color_, label_, alpha_ in \
                izip_with_cycles(pairs, dither, marker, s, color, label, alpha):

            rho0 = scipy.stats.spearmanr(xs, ys)[0]
            rho1 = scipy.stats.spearmanr(ys, xs)[0]
            if not np.isclose(rho0, rho1):
                # should never happen
                raise RuntimeError("Error calculating Spearman's rho")
            ax.annotate('$\\rho=%.3f$' % rho0, (0.05, 0.9), xycoords='axes fraction')

            if dither_ is not None:
                xs = np.random.normal(xs, dither_)
                ys = np.random.normal(ys, dither_)
            ax.scatter(xs, ys, marker=marker_, s=s_, color=color_,
                       alpha=alpha_, label=label_, **kwargs)

        if label:
            legend = ax.legend(loc='upper left', markerscale=80, scatterpoints=1)
            for lbl in legend.get_texts():
                lbl.set_fontsize('small')

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)
        if save_to is None:
            fig.show()
        else:
            fig.savefig(save_to)
            plt.close(fig)


if __name__ == "__main__":
    run(parse_args())
