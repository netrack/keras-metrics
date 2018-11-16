import keras
import keras.backend
import keras_metrics
import numpy
import unittest


class TestMetrics(unittest.TestCase):

    def test_metrics(self):
        tp = keras_metrics.true_positive()
        tn = keras_metrics.true_negative()
        fp = keras_metrics.false_positive()
        fn = keras_metrics.false_negative()

        precision = keras_metrics.precision()
        recall = keras_metrics.recall()
        f1 = keras_metrics.f1_score()

        model = keras.models.Sequential()
        model.add(keras.layers.Activation(keras.backend.sin))
        model.add(keras.layers.Activation(keras.backend.abs))

        model.compile(optimizer="sgd",
                      loss="binary_crossentropy",
                      metrics=[tp, tn, fp, fn, precision, recall, f1])

        samples = 10000
        batch_size = 100
        lim = numpy.pi/2

        x = numpy.random.uniform(0, lim, (samples, 1))
        y = numpy.random.randint(2, size=(samples, 1))

        model.fit(x, y, epochs=10, batch_size=batch_size)
        metrics = model.evaluate(x, y, batch_size=batch_size)[1:]

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

        self.assertAlmostEqual(expected_precision, precision)
        self.assertAlmostEqual(expected_recall, recall)
        self.assertAlmostEqual(expected_f1, f1)


if __name__ == "__main__":
    unittest.main()
