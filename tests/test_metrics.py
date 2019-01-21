import keras
import keras.backend
import keras_metrics
import itertools
import numpy
import tempfile
import unittest


class TestMetrics(unittest.TestCase):

    def setUp(self):
        tp = keras_metrics.true_positive()
        tn = keras_metrics.true_negative()
        fp = keras_metrics.false_positive()
        fn = keras_metrics.false_negative()

        precision = keras_metrics.precision()
        recall = keras_metrics.recall()
        f1 = keras_metrics.f1_score()

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Activation(keras.backend.sin))
        self.model.add(keras.layers.Activation(keras.backend.abs))

        self.model.compile(optimizer="sgd",
                           loss="binary_crossentropy",
                           metrics=[tp, tn, fp, fn, precision, recall, f1])

    def samples(self, n):
        x = numpy.random.uniform(0, numpy.pi/2, (n, 1))
        y = numpy.random.randint(2, size=(n, 1))
        return x, y

    def test_save_load(self):
        custom_objects = {
            "true_positive": keras_metrics.true_positive(),
            "true_negative": keras_metrics.true_negative(),
            "false_positive": keras_metrics.false_positive(),
            "false_negative": keras_metrics.false_negative(),
            "precision": keras_metrics.precision(),
            "recall": keras_metrics.recall(),
            "f1_score": keras_metrics.f1_score(),
            "sin": keras.backend.sin,
            "abs": keras.backend.abs,
        }

        x, y = self.samples(100)
        self.model.fit(x, y, epochs=10)

        with tempfile.NamedTemporaryFile() as file:
            self.model.save(file.name, overwrite=True)
            model = keras.models.load_model(file.name, custom_objects=custom_objects)

            expected = self.model.evaluate(x, y)[1:]
            received = model.evaluate(x, y)[1:]

            self.assertEqual(expected, received)

    def test_metrics(self):
        samples = 10000
        batch_size = 100

        x, y = self.samples(samples)

        self.model.fit(x, y, epochs=10, batch_size=batch_size)
        metrics = self.model.evaluate(x, y, batch_size=batch_size)[1:]

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


if __name__ == "__main__":
    unittest.main()
