import keras
import keras.utils
import keras_metrics as km
import numpy
import unittest


class TestAverageRecall(unittest.TestCase):

    def create_samples(self, n, labels=1):
        x = numpy.random.uniform(0, numpy.pi/2, (n, labels))
        y = numpy.random.randint(labels, size=(n, 1))
        return x, keras.utils.to_categorical(y)

    def test_average_recall(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Activation(keras.backend.sin))
        model.add(keras.layers.Activation(keras.backend.abs))
        model.add(keras.layers.Softmax())
        model.compile(optimizer="sgd",
                      loss="categorical_crossentropy",
                      metrics=[
                          km.categorical_recall(label=0),
                          km.categorical_recall(label=1),
                          km.categorical_recall(label=2),
                          km.categorical_average_recall(labels=3),
                      ])

        x, y = self.create_samples(10000, labels=3)

        model.fit(x, y, epochs=10, batch_size=100)
        metrics = model.evaluate(x, y, batch_size=100)[1:]

        r0, r1, r2 = metrics[0:3]
        average_recall = metrics[3]

        expected_recall = (r0+r1+r2)/3.0
        self.assertAlmostEqual(expected_recall, average_recall, places=3)


if __name__ == "__main__":
    unittest.main()
