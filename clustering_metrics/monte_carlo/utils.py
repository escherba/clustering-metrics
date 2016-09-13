from clustering_metrics.metrics import ClusteringMetrics, ConfusionMatrix2
from pymaptools.inspect import iter_method_names


CONTINGENCY_METRICS = list(iter_method_names(ClusteringMetrics))
PAIRWISE_METRICS = list(iter_method_names(ConfusionMatrix2))

BENCHMARKS = ['time_cpu']

INCIDENCE_METRICS = PAIRWISE_METRICS + CONTINGENCY_METRICS

ROC_METRICS = ['roc_max_info', 'roc_auc']
LIFT_METRICS = ['aul_score']

RANKING_METRICS = ROC_METRICS + LIFT_METRICS

METRICS = RANKING_METRICS + INCIDENCE_METRICS + BENCHMARKS


def serialize_args(args):
    namespace = dict(args.__dict__)
    fields_to_delete = ["input", "output", "func", "logging"]
    for field in fields_to_delete:
        try:
            del namespace[field]
        except KeyError:
            pass
    return namespace
