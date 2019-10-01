"""
MNIST dataset example
"""
import numpy as np
from tensorflow.keras import datasets

from barrage import BarrageModel
from barrage.utils import io_utils


def get_data():
    """Load MNIST dataset."""
    (X_train, y_train), (X_val, y_val) = datasets.mnist.load_data()
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


if __name__ == "__main__":
    records_train, records_val = get_data()
    # Train
    cfg = io_utils.load_json("config_mnist.json")
    BarrageModel("artifacts").train(cfg, records_train, records_val)
