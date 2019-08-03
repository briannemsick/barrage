"""
MNIST dataset example
"""
import numpy as np
from tensorflow.keras import layers, models

from barrage import BarrageModel
from barrage.utils import io_utils


def get_data():
    """Load MNIST dataset."""
    from keras.datasets import mnist

    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    X_train = X_train[:, ..., np.newaxis]  # need image shape (28, 28, 1) not (28, 28)
    X_val = X_val[:, ..., np.newaxis]  # need image shape (28, 28, 1) not (28, 28)

    # Convert to list of dicts
    samples_train = X_train.shape[0]
    records_train = [
        {"x": X_train[ii, ...], "y": y_train[ii]} for ii in range(samples_train)
    ]
    samples_val = X_val.shape[0]
    records_val = [{"x": X_val[ii, ...], "y": y_val[ii]} for ii in range(samples_val)]
    return records_train, records_val


def net():
    """Simple MNIST CNN.

    Note: we could have used barrage.model.sequential_from_config.
    """
    inputs = layers.Input(shape=(28, 28, 1), name="img")
    conv_1 = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    conv_2 = layers.Conv2D(64, (3, 3), activation="relu")(conv_1)
    mp = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)
    flatten = layers.Flatten()(mp)
    dense = layers.Dense(128, activation="relu")(flatten)
    outputs = layers.Dense(10, activation="softmax", name="target")(dense)
    return models.Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    records_train, records_val = get_data()
    # Train
    cfg = io_utils.load_json("config_mnist.json")
    BarrageModel("artifacts").train(cfg, records_train, records_val)
