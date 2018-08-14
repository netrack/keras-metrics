# Keras Metrics

[![Build Status][BuildStatus]](https://travis-ci.org/netrack/keras-metrics)

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
```py
import keras
import keras_metrics

model = models.Sequential()
model.add(keras.layers.Dense(1, activation="sigmoid", input_dim=2))
model.add(keras.layers.Dense(1, activation="softmax"))

model.compile(optimizer="sgd",
              loss="binary_crossentropy",
              metrics=[keras_metrics.precision(), keras_metrics.recall()])
```

Similar configuration for multi-label binary crossentropy:
```py
import keras
import keras_metrics

model = models.Sequential()
model.add(keras.layers.Dense(1, activation="sigmoid", input_dim=2))
model.add(keras.layers.Dense(2, activation="softmax"))

# Calculate precision for the second label.
precision = keras_metrics.precision(label=1)

# Calculate recall for the first label.
recall = keras_metrics.recall(label=0)

model.compile(optimizer="sgd",
              loss="binary_crossentropy",
              metrics=[precision, recall])
```

[BuildStatus]: https://travis-ci.org/netrack/keras-metrics.svg?branch=master
