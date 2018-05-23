from keras import backend as K
from keras.layers import Layer
from operator import truediv


def _int32(y_true, y_pred):
    y_true = K.cast(y_true, "int32")
    y_pred = K.cast(K.round(y_pred), "int32")
    return y_true, y_pred


class true_positive(Layer):
    """Create a metric for model's true positives amount calculation.

    A true positive is an outcome where the model correctly predicts the
    positive class.
    """

    def __init__(self, name="true_positive", **kwargs):
        super(true_positive, self).__init__(name=name, **kwargs)
        self.tp = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.tp, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _int32(y_true, y_pred)

        tp = K.sum(y_true * y_pred)
        current_tp = self.tp * 1

        tp_update = K.update_add(self.tp, tp)
        self.add_update(tp_update, inputs=[y_true, y_pred])

        return tp + current_tp


class true_negative(Layer):
    """Create a metric for model's true negatives amount calculation.

    A true negative is an outcome where the model correctly predicts the
    negative class.
    """

    def __init__(self, name="true_negative", **kwargs):
        super(true_negative, self).__init__(name=name, **kwargs)
        self.tn = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.tn, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _int32(y_true, y_pred)

        neg_y_true = 1 - y_true
        neg_y_pred = 1 - y_pred

        tn = K.sum(neg_y_true * neg_y_pred)
        current_tn = self.tn * 1

        tn_update = K.update_add(self.tn, tn)
        self.add_update(tn_update, inputs=[y_true, y_pred])

        return tn + current_tn


class false_negative(Layer):
    """Create a metric for model's false negatives amount calculation.

    A false negative is an outcome where the model incorrectly predicts the
    negative class.
    """

    def __init__(self, name="false_negative", **kwargs):
        super(false_negative, self).__init__(name=name, **kwargs)
        self.fn = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.fn, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _int32(y_true, y_pred)
        neg_y_pred = 1 - y_pred

        fn = K.sum(y_true * neg_y_pred)
        current_fn = self.fn * 1

        fn_update = K.update_add(self.fn, fn)
        self.add_update(fn_update, inputs=[y_true, y_pred])

        return fn + current_fn


class false_positive(Layer):
    """Create a metric for model's false positive amount calculation.

    A false positive is an outcome where the model incorrectly predicts the
    positive class.
    """

    def __init__(self, name="false_positive", **kwargs):
        super(false_positive, self).__init__(name=name, **kwargs)
        self.fp = K.variable(0, dtype="int32")

    def reset_states(self):
        """Reset the state of the metric."""
        K.set_value(self.fp, 0)

    def __call__(self, y_true, y_pred):
        y_true, y_pred = _int32(y_true, y_pred)
        neg_y_true = 1 - y_true

        fp = K.sum(neg_y_true * y_pred)
        current_fp = self.fp * 1

        fp_update = K.update_add(self.fp, fp)
        self.add_update(fp_update, inputs=[y_true, y_pred])

        return fp + current_fp


class recall(Layer):
    """Create a metric for model's recall calculation.

    Recall measures propotion of actual positives that was indetified correctly.
    """

    def __init__(self, name="recall", **kwargs):
        super(recall, self).__init__(name=name, **kwargs)

        self.tp = true_positive()
        self.fn = false_negative()
        self.capping = K.constant(1, dtype="int32")

    def reset_states(self):
        """Reset the state of the metrics."""
        self.tp.reset_states()
        self.fn.reset_states()

    def __call__(self, y_true, y_pred):
        tp = self.tp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)

        div = K.maximum((tp + fn), self.capping)
        return truediv(tp, div)


class precision(Layer):
    """Create  a metric for mode's precision calculation.

    Precision measures proportion of positives identifications that were
    actually correct.
    """

    def __init__(self, name="precision", **kwargs):
        super(precision, self).__init__(name=name, **kwargs)

        self.tp = true_positive()
        self.fp = false_positive()
        self.capping = K.constant(1, dtype="int32")

    def reset_states(self):
        """Reset the state of the metrics."""
        self.tp.reset_states()
        self.fp.reset_states()

    def __call__(self, y_true, y_pred):
        tp = self.tp(y_true, y_pred)
        fp = self.fp(y_true, y_pred)

        div = K.maximum((tp + fp), self.capping)
        return truediv(tp, div)
