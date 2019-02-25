import keras
import keras.backend
import keras.utils
import keras.regularizers
import keras_metrics as km
import itertools
import numpy
import tempfile
import unittest

from keras import backend as K


class TestMetrics(unittest.TestCase):

    binary_metrics = [
        km.binary_true_positive,
        km.binary_true_negative,
        km.binary_false_positive,
        km.binary_false_negative,
        km.binary_precision,
        km.binary_recall,
        km.binary_f1_score,
    ]

    categorical_metrics = [
        km.categorical_true_positive,
        km.categorical_true_negative,
        km.categorical_false_positive,
        km.categorical_false_negative,
        km.categorical_precision,
        km.categorical_recall,
        km.categorical_f1_score,
    ]

    sparse_categorical_metrics = [
        km.sparse_categorical_true_positive,
        km.sparse_categorical_true_negative,
        km.sparse_categorical_false_positive,
        km.sparse_categorical_false_negative,
        km.sparse_categorical_precision,
        km.sparse_categorical_recall,
        km.sparse_categorical_f1_score,
    ]

    def create_binary_samples(self, n):
        x = numpy.random.uniform(0, numpy.pi/2, (n, 1))
        y = numpy.random.randint(2, size=(n, 1))
        return x, y

    def create_categorical_samples(self, n):
        x, y = self.create_binary_samples(n)
        return x, keras.utils.to_categorical(y)

    def create_metrics(self, metrics_fns):
        return list(map(lambda m: m(), metrics_fns))

    def create_model(self, outputs, loss, metrics_fns):
        model = keras.models.Sequential()
        model.add(keras.layers.Activation(keras.backend.sin))
        model.add(keras.layers.Activation(keras.backend.abs))
        model.add(keras.layers.Lambda(lambda x: K.concatenate([x]*outputs)))
        model.compile(optimizer="sgd",
                      loss=loss,
                      metrics=self.create_metrics(metrics_fns))
        return model

    def assert_save_load(self, model, metrics_fns, samples_fn):
        metrics = [m() for m in metrics_fns]

        custom_objects = {m.__name__: m for m in metrics}
        custom_objects["sin"] = keras.backend.sin
        custom_objects["abs"] = keras.backend.abs

        x, y = samples_fn(100)
        model.fit(x, y, epochs=10)

        with tempfile.NamedTemporaryFile() as file:
            model.save(file.name, overwrite=True)

            loaded_model = keras.models.load_model(
                file.name, custom_objects=custom_objects)

            expected = model.evaluate(x, y)[1:]
            received = loaded_model.evaluate(x, y)[1:]

            self.assertEqual(expected, received)

    def test_save_load_binary_metrics(self):
        model = self.create_model(1, "binary_crossentropy",
                                  self.binary_metrics)
        self.assert_save_load(model,
                              self.binary_metrics,
                              self.create_binary_samples)

    def test_save_load_categorical_metrics(self):
        model = self.create_model(2, "categorical_crossentropy",
                                  self.categorical_metrics)
        self.assert_save_load(model,
                              self.categorical_metrics,
                              self.create_categorical_samples)

    def test_save_load_sparse_categorical_metrics(self):
        model = self.create_model(2, "sparse_categorical_crossentropy",
                                  self.sparse_categorical_metrics)
        self.assert_save_load(model,
                              self.sparse_categorical_metrics,
                              self.create_binary_samples)

    def assert_metrics(self, model, samples_fn):
        samples = 10000
        batch_size = 100

        x, y = samples_fn(samples)

        model.fit(x, y, epochs=10, batch_size=batch_size)
        metrics = model.evaluate(x, y, batch_size=batch_size)[1:]

        metrics = list(map(float, metrics))

        tp_val = metrics[0]
        tn_val = metrics[1]
        fp_val = metrics[2]
        fn_val = metrics[3]

        precision = metrics[4]
        recall = metrics[5]
        f1 = metrics[6]

        expected_precision = tp_val / (tp_val + fp_val)
        expected_recall = tp_val / (tp_val + fn_val)

        f1_divident = (expected_precision*expected_recall)
        f1_divisor = (expected_precision+expected_recall)
        expected_f1 = (2 * f1_divident / f1_divisor)

        self.assertGreaterEqual(tp_val, 0.0)
        self.assertGreaterEqual(fp_val, 0.0)
        self.assertGreaterEqual(fn_val, 0.0)
        self.assertGreaterEqual(tn_val, 0.0)

        self.assertEqual(sum(metrics[0:4]), samples)

        places = 4
        self.assertAlmostEqual(expected_precision, precision, places=places)
        self.assertAlmostEqual(expected_recall, recall, places=places)
        self.assertAlmostEqual(expected_f1, f1, places=places)

    def test_binary_metrics(self):
        model = self.create_model(1, "binary_crossentropy",
                                  self.binary_metrics)
        self.assert_metrics(model, self.create_binary_samples)

    def test_categorical_metrics(self):
        model = self.create_model(2, "categorical_crossentropy",
                                  self.categorical_metrics)
        self.assert_metrics(model, self.create_categorical_samples)

    def test_sparse_categorical_metrics(self):
        model = self.create_model(2, "sparse_categorical_crossentropy",
                                  self.sparse_categorical_metrics)
        self.assert_metrics(model, self.create_binary_samples)


if __name__ == "__main__":
    unittest.main()
