from functools import partial
from keras_metrics import metrics as m
from keras_metrics import casts


__version__ = "1.1.0"


def metric_fn(cls, cast_strategy):
    def fn(label=0, **kwargs):
        metric = cls(label=label, cast_strategy=cast_strategy, **kwargs)
        metric.__name__ = "%s_%s" % (cast_strategy.__name__, cls.__name__)
        return metric
    return fn


binary_metric = partial(
    metric_fn, cast_strategy=casts.binary)


categorical_metric = partial(
    metric_fn, cast_strategy=casts.categorical)


sparse_categorical_metric = partial(
    metric_fn, cast_strategy=casts.sparse_categorical)


binary_true_positive = binary_metric(m.true_positive)
binary_true_negative = binary_metric(m.true_negative)
binary_false_positive = binary_metric(m.false_positive)
binary_false_negative = binary_metric(m.false_negative)
binary_precision = binary_metric(m.precision)
binary_recall = binary_metric(m.recall)
binary_f1_score = binary_metric(m.f1_score)
binary_average_recall = binary_metric(m.average_recall)


categorical_true_positive = categorical_metric(m.true_positive)
categorical_true_negative = categorical_metric(m.true_negative)
categorical_false_positive = categorical_metric(m.false_positive)
categorical_false_negative = categorical_metric(m.false_negative)
categorical_precision = categorical_metric(m.precision)
categorical_recall = categorical_metric(m.recall)
categorical_f1_score = categorical_metric(m.f1_score)
categorical_average_recall = categorical_metric(m.average_recall)


sparse_categorical_true_positive = sparse_categorical_metric(m.true_positive)
sparse_categorical_true_negative = sparse_categorical_metric(m.true_negative)
sparse_categorical_false_positive = sparse_categorical_metric(m.false_positive)
sparse_categorical_false_negative = sparse_categorical_metric(m.false_negative)
sparse_categorical_precision = sparse_categorical_metric(m.precision)
sparse_categorical_recall = sparse_categorical_metric(m.recall)
sparse_categorical_f1_score = sparse_categorical_metric(m.f1_score)
sparse_categorical_average_recall = sparse_categorical_metric(m.average_recall)


# For backward compatibility.
true_positive = binary_true_positive
true_negative = binary_true_negative
false_positive = binary_false_positive
false_negative = binary_false_negative
precision = binary_precision
recall = binary_recall
f1_score = binary_f1_score
