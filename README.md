# Keras Metrics

This package provides metrics for evaluation of Keras classification models.
The metrics are safe to use for batch-based model evaluation.

## Installation

To install the package from the PyPi repository you can execute the following
command:
```sh
pip install keras-metrics
```

## Usage

The usage of the package is simple:
```python
import keras
import keras_metrics

model = models.Sequential()
model.add(keras.layers.Dense(1, activation="sigmoid", input_dim=2))
model.add(keras.layers.Dense(1, activation="softmax"))

model.compile(optimizer="sgd",
              loss="binary_crossentropy",
              metrics=[keras_metrics.precision(), keras_metrics.recall()])
```
