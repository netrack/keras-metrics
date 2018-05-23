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

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1, activation="sigmoid", input_dim=2))
        model.add(keras.layers.Dense(1, activation="softmax"))

        model.compile(optimizer="sgd",
                      loss="binary_crossentropy",
                      metrics=[tp, fp, fn, precision, recall])

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

        expected_precision = tp_val / (tp_val + fp_val)
        expected_recall = tp_val / (tp_val + fn_val)

        self.assertAlmostEqual(expected_precision, precision, delta=0.05)
        self.assertAlmostEqual(expected_recall, recall, delta=0.05)


if __name__ == "__main__":
    unittest.main()
