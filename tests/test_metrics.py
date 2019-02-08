import keras
import keras.backend
import keras_metrics
import itertools
import numpy
import tempfile
import unittest


class TestMetrics(unittest.TestCase):

    def __init__(self, methodName, sparse=False):
        super(TestMetrics, self).__init__(methodName=methodName)
        self.sparse = sparse

    def setUp(self):
        tp = keras_metrics.true_positive(sparse=self.sparse)
        tn = keras_metrics.true_negative(sparse=self.sparse)
        fp = keras_metrics.false_positive(sparse=self.sparse)
        fn = keras_metrics.false_negative(sparse=self.sparse)

        precision = keras_metrics.precision(sparse=self.sparse)
        recall = keras_metrics.recall(sparse=self.sparse)
        f1 = keras_metrics.f1_score(sparse=self.sparse)

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Activation(keras.backend.sin))
        self.model.add(keras.layers.Activation(keras.backend.abs))

        if self.sparse:
            loss = "sparse_categorical_crossentropy"
        else:
            loss = "binary_crossentropy"

        self.model.compile(optimizer="sgd",
                           loss=loss,
                           metrics=[tp, tn, fp, fn, precision, recall, f1])

    def samples(self, n):
        if self.sparse:
            categories = 2
            x = numpy.random.uniform(0, numpy.pi/2, (n, categories))
            y = numpy.random.randint(categories, size=(n, 1))
        else:
            x = numpy.random.uniform(0, numpy.pi/2, (n, 1))
            y = numpy.random.randint(2, size=(n, 1))
        return x, y

    def test_save_load(self):
        custom_objects = {
            "true_positive": keras_metrics.true_positive(sparse=self.sparse),
            "true_negative": keras_metrics.true_negative(sparse=self.sparse),
            "false_positive": keras_metrics.false_positive(sparse=self.sparse),
            "false_negative": keras_metrics.false_negative(sparse=self.sparse),
            "precision": keras_metrics.precision(sparse=self.sparse),
            "recall": keras_metrics.recall(sparse=self.sparse),
            "f1_score": keras_metrics.f1_score(sparse=self.sparse),
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


def suite():
    s = unittest.TestSuite()
    s.addTests(TestMetrics(methodName=method, sparse=sparse)
               for method in unittest.defaultTestLoader.getTestCaseNames(TestMetrics)
               for sparse in (False, True))
    return s


if __name__ == "__main__":
    import sys
    result = unittest.TextTestRunner().run(suite())
    sys.exit(not result.wasSuccessful())
    # unittest.main()
