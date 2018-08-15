import keras
import keras_metrics
import numpy
import unittest


class TestMetrics(unittest.TestCase):

    def test_metrics(self):
        tp = keras_metrics.true_positive()
        fp = keras_metrics.false_positive()
        fn = keras_metrics.false_negative()

        precision = keras_metrics.precision()
        recall = keras_metrics.recall()
        f1 = keras_metrics.f1_score()

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1, activation="sigmoid", input_dim=2))
        model.add(keras.layers.Dense(1, activation="softmax"))

        model.compile(optimizer="sgd",
                      loss="binary_crossentropy",
                      metrics=[tp, fp, fn, precision, recall, f1])

        samples = 1000
        x = numpy.random.random((samples, 2))
        y = numpy.random.randint(2, size=(samples, 1))

        model.fit(x, y, epochs=1, batch_size=10)
        metrics = model.evaluate(x, y, batch_size=10)[1:]

        tp_val = metrics[0]
        fp_val = metrics[1]
        fn_val = metrics[2]

        precision = metrics[3]
        recall = metrics[4]
        f1 = metrics[5]

        expected_precision = tp_val / (tp_val + fp_val)
        expected_recall = tp_val / (tp_val + fn_val)

        f1_divident = (expected_precision*expected_recall)
        f1_divisor = (expected_precision+expected_recall)
        expected_f1 = (2 * f1_divident / f1_divisor)

        self.assertAlmostEqual(expected_precision, precision, delta=0.05)
        self.assertAlmostEqual(expected_recall, recall, delta=0.05)
        self.assertAlmostEqual(expected_f1, f1, delta=0.05)


if __name__ == "__main__":
    unittest.main()
