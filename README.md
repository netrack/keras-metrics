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
import keras_metrics as km

model = models.Sequential()
model.add(keras.layers.Dense(1, activation="sigmoid", input_dim=2))
model.add(keras.layers.Dense(1, activation="softmax"))

model.compile(optimizer="sgd",
              loss="binary_crossentropy",
              metrics=[km.binary_precision(), km.binary_recall()])
```

Similar configuration for multi-label binary crossentropy:
```py
import keras
import keras_metrics as km

model = models.Sequential()
model.add(keras.layers.Dense(1, activation="sigmoid", input_dim=2))
model.add(keras.layers.Dense(2, activation="softmax"))

# Calculate precision for the second label.
precision = km.binary_precision(label=1)

# Calculate recall for the first label.
recall = km.binary_recall(label=0)

model.compile(optimizer="sgd",
              loss="binary_crossentropy",
              metrics=[precision, recall])
```

Keras metrics package also supports metrics for categorical crossentropy and
sparse categorical crossentropy:
```py
import keras_metrics as km

c_precision = km.categorical_precision()
sc_precision = km.sparse_categorical_precision()

# ...
```

## Tensorflow Keras

Tensorflow library provides the ```keras``` package as parts of its API, in
order to use ```keras_metrics``` with Tensorflow Keras, you are advised to
perform model training with initialized global variables:
```py
import numpy as np
import keras_metrics as km
import tensorflow as tf
import tensorflow.keras as keras

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation="softmax"))
model.compile(optimizer="sgd",
              loss="binary_crossentropy",
              metrics=[km.binary_true_positive()])

x = np.array([[0], [1], [0], [1]])
y = np.array([1, 0, 1, 0]

# Wrap model.fit into the session with global
# variables initialization.
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    model.fit(x=x, y=y)
```

[BuildStatus]: https://travis-ci.org/netrack/keras-metrics.svg?branch=master
