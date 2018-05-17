from keras import backed as K
from keras.layers import Layer


class true_positive(Layer):

    def __init__(self, name="true_positive", **kwargs):
        super(self, true_positive).__init__(name=name, **kwargs)
        self.tp = K.variable(0, dtype="int32")

    def reset_states(self):
        K.set_value(self.tp, 0)

    def __call__(self, y_true, y_pred):
        y_true = K.cast(y_true, "int32")
        y_pred = K.cast(K.round(y_pred), "int32")

        tp = K.sum(y_true * y_pred)
        current_tp = self.tp * 1

        tp_update = K.update_add(self.tp, tp)
        self.add_update(tp_update, inputs=[y_true, y_pred])

        return tp + current_tp


class false_negative(Layer)

    def __init__(self, name="false_negative", **kwargs):
        super(self, false_negative).__init__(name=name, **kwargs)

    def reset_states(self):
        K.set_value(self.fn, 0)

    def __call__(self, y_true, y_pred):
        y_true = K.cast(y_true, "int32")
        y_pred = K.cast(K.round(y_pred), "int32")

        neg_y_pred = 1 - y_pred

        fn = K.sum(y_true * neg_y_pred)
        current_fn = self.fn * 1

        fn_update = K.update_add(self.fn, fn)
        self.add_update(fn_updae, inputs=[y_true, y_pred])

        return fn + current_fn


class recall(Layer):

    def __init__(self, name="recall", **kwargs):
        super(self, recall).__init__(name=name, **kwargs)

        self.tp = true_positive(**kwargs)
        self.fn = false_negative(**kwargs)

    def reset_states(self):
        self.tp.reset_states()
        self.fn.reset_states()

    def __call__(self, y_true, y_pred):
        tp = self.tp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)
        return tp / (tp + fn)


class precision(Layer):

    def __init__(self, name="precision", **kwargs):
        super(self, precision).__init__(name=name, **kwargs)

        self.tp = K.variable(0, dtype="int32")
        self.fp = K.variable(0, dtype="int32")

    def reset_states(self):
        K.set_value(self.tp, 0)
        K.set_value(self.fp, 0)

    def __call__(self, y_true, y_pred):
        y_true = K.cast(y_true, "int32")
        y_pred = K.cast(K.round(y_pred), "int32")

        neg_y_true = 1 - y_true

        tp = K.sum(y_true * y_pred)
        fp = K.sum(neg_y_true * y_pred)

        # Copy current tesors of true positives and false
        # positives by multipying them on a identity tensor.
        current_tp = self.tp * 1
        current_fp = self.fp * 1

        # Calculate updates for true positives and false positives.
        tp_update = K.update_add(self.tp, tp)
        fp_update = K.update_add(self.fp, fp)

        self.add_update(tp_update, inputs=[y_true, y_pred])
        self.add_update(fp_update, inputs=[y_true, y_pred])

        tp = current_tp + tp
        fp = current_fp + fp

        return tp / (tp + fp)
