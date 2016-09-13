"""
To benchmark the proposed implementation vs the one in Scikit-Learn:

::

    ipython scripts/benchmark.py -- --method ami --implementation sklearn
    ipython scripts/benchmark.py -- --method ami --implementation proposed

"""
import os
import sys
import argparse
import numpy as np
import cPickle as pickle
from IPython import get_ipython


METHODS = {
    'hcv': ('homogeneity_completeness_v_measure', 'entropy_scores'),
    'ami': ('adjusted_mutual_info_score', 'adjusted_mutual_info'),
    'ari': ('adjusted_rand_score', 'adjusted_rand_index')
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', type=str, default='proposed',
                        choices=['sklearn', 'proposed', 'oo'],
                        help='which implementation to benchmark')
    parser.add_argument('--method', type=str, help='method to benchmark',
                        default='ami',
                        choices=METHODS.keys())
    parser.add_argument('--num_tests', type=int, default=3,
                        help='how many tests to run')
    parser.add_argument('--max_classes', type=int, default=500,
                        help='maximum number of classes')
    parser.add_argument('--max_clusters', type=int, default=500,
                        help='maximum number of clusters')
    parser.add_argument('--num_samples', type=int, default=20000,
                        help='sample size (number of labels)')
    namespace = parser.parse_args(args)
    return namespace


ARGS = parse_args()


ipython = get_ipython()
if ipython is None:
    print "You should run this script with ``ipython`` interpreter"
    sys.exit(0)


PATH = "out-c%d-k%d-s%d.pickle" % (
    ARGS.max_classes, ARGS.max_clusters, ARGS.num_samples)


if os.path.exists(PATH):
    print "Loading data from %s" % PATH
    with open(PATH, 'r') as fh:
        ltrue, lpred = pickle.load(fh)
else:
    shape = (ARGS.num_samples,)
    ltrue = np.random.randint(low=0, high=ARGS.max_classes, size=shape)
    lpred = np.random.randint(low=0, high=ARGS.max_clusters, size=shape)
    print "Saving generated data to %s" % PATH
    with open(PATH, 'w') as fh:
        pickle.dump((ltrue, lpred), fh, protocol=pickle.HIGHEST_PROTOCOL)


if ARGS.implementation == 'oo':
    from clustering_metrics.metrics import ClusteringMetrics
    cm = ClusteringMetrics.from_labels(ltrue, lpred)
    method = getattr(cm, METHODS[ARGS.method][1])
    line = "method()"
elif ARGS.implementation == 'sklearn':
    import sklearn.metrics.cluster as module
    method = getattr(module, METHODS[ARGS.method][0])
    line = "method(ltrue, lpred)"
elif ARGS.implementation == 'proposed':
    import clustering_metrics.metrics as module
    method = getattr(module, METHODS[ARGS.method][0])
    line = "method(ltrue, lpred)"
else:
    raise argparse.ArgumentError('Unknown value for --implementation')


print "Sanity check:"
print "\t{} = {}".format(ARGS.method, eval(line))

for idx in xrange(ARGS.num_tests):
    print "Running test {}/{}...".format(idx + 1, ARGS.num_tests)
    ipython.magic("timeit " + line)
