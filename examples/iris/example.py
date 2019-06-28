"""
Iris dataset example.

This example demonstrates the following:

    1. Simple "vanilla" example - how to setup a basic config on an easy dataset.

    2. "overkill" example - how to write a custom loader, transformer, and augmentation
    function.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import layers, models

from barrage import BarrageModel
from barrage.dataset import RecordMode, RecordLoader, RecordTransformer
from barrage.utils import io_utils


def get_data():
    """Load iris dataset."""
    dataset = load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    records_train = pd.DataFrame(X_train, columns=["i1", "i2", "i3", "i4"])
    records_train["label"] = y_train

    records_val = pd.DataFrame(X_val, columns=["i1", "i2", "i3", "i4"])
    records_val["label"] = y_val

    return records_train, records_val


def net(num_dense=2, dim_dense=10):
    """Simple dense network."""

    # Take note of the input name for the config
    inputs = layers.Input(shape=(4,), name="iris")
    dense = inputs

    for _ in range(num_dense):
        dense = layers.Dense(dim_dense, activation="relu")(dense)

    # Take note of the output name for the config
    outputs = layers.Dense(3, activation="softmax", name="flower")(dense)

    return models.Model(inputs=inputs, outputs=outputs)


def vanilla_iris():
    """Here we use sklearn to mean variance normalize the dataset and train a
    simple model.
    """
    # Get data
    records_train, records_val = get_data()

    # For now we will use sklearn.preprocessing.StandardScaler because the dataset
    # fits in memory. However, this approach does not scale and the overkill example
    # will demonstrate how to apply mean-var normalizaton with a dataset that does not
    # fit into memory.
    cols = ["i1", "i2", "i3", "i4"]
    scaler = StandardScaler().fit(records_train[cols])
    records_train[cols] = scaler.transform(records_train[cols])
    records_val[cols] = scaler.transform(records_val[cols])
    scaler_params = {"mean": scaler.mean_, "std": scaler.scale_}
    print(f"scaler params {scaler_params}")

    # Specify config
    cfg = {
        "dataset": {
            "loader": {
                # use built in KeySelector
                "import": "KeySelector",
                "params": {
                    "inputs": {
                        # name matches 'inputs' name
                        "iris": ["i1", "i2", "i3", "i4"]
                    },
                    "outputs": {
                        # name matches 'outputs' name
                        "flower": ["label"]
                    },
                },
            },
            "seed": 42,
        },
        "model": {
            # use the net we defined
            "network": {
                "import": "example.net",
                "params": {"num_dense": 4, "dim_dense": 25},
            },
            "outputs": [
                {
                    # name matches 'outputs' name
                    "name": "flower",
                    "loss": {"import": "sparse_categorical_crossentropy"},
                    "metrics": [{"import": "accuracy"}],
                }
            ],
        }
        # use defaults for solver, services
    }

    # Train the model
    BarrageModel("artifacts_vanilla").train(cfg, records_train, records_val)


def overkill_iris():
    """Here we use a custom loader, transformer, and augmentation functions."""

    # Get data
    records_train, records_val = get_data()

    # Specify config
    cfg = {
        "dataset": {
            # use custom loader
            "loader": {"import": "example.CustomIrisLoader"},
            "transformer": {
                "import": "example.CustomInputMeanVarTransformer",
                "params": {"key": "iris"},
            },
            "augmentor": [
                {
                    "import": "example.add_input_noise",
                    "params": {"key": "iris", "scale": 0.1},
                }
            ],
            "seed": 42,
        },
        "model": {
            "network": {
                "import": "example.net",
                "params": {"num_dense": 4, "dim_dense": 25},
            },
            "outputs": [
                {
                    "name": "flower",
                    "loss": {"import": "sparse_categorical_crossentropy"},
                    "metrics": [{"import": "accuracy"}],
                }
            ],
        },
        # specify solver
        "solver": {
            "optimizer": {"import": "Adam", "learning_rate": 1e-3},
            "batch_size": 32,
            "epochs": 50,
        },
        # choose best model based on 'val_accuracy'
        "services": {"best_checkpoint": {"monitor": "val_accuracy", "mode": "max"}},
    }

    # Train the model
    BarrageModel("artifacts_overkill").train(cfg, records_train, records_val)


class CustomIrisLoader(RecordLoader):
    def load(self, record):

        # The data is stored directly inside the DataFrame - we can directly index.
        # The KeySelector is more general in this situation and the prefered way
        # to do things. That being said, let's hardcode a loader. For the transformer
        # and augmentation functions we will be more general and could be reused on
        # other datasets.

        # Note: if the data was a filepath we could load the file here instead.
        keys = ["i1", "i2", "i3", "i4"]
        X = {"iris": np.array([record[k] for k in keys])}
        if self.mode == RecordMode.TRAIN or self.mode == RecordMode.VALIDATION:
            y = {"flower": np.array(record["label"])}
            return (X, y)
        else:
            return (X,)


class CustomInputMeanVarTransformer(RecordTransformer):
    def fit(self, records):
        # Records underlying data fits in memory for iris. But if we pretended it did
        # not, we can still make this transformer function by using the loader and
        # iterating over each record
        key = self.params["key"]

        # We will use Var(X - K) = Var(X) in computation
        data_record_0 = self.loader.load(records[0])

        # Reminder data record is (X, y) or (X, y, w)
        K = data_record_0[0][key]

        # Used in variance computation
        mean = np.zeros_like(K)
        ex = np.zeros_like(K)
        ex2 = np.zeros_like(K)
        n = len(records)

        # Iterate over records
        for ii in range(len(records)):
            # Load record
            dr = self.loader.load(records[ii])
            X = dr[0][key]

            mean += X
            ex += X - K
            ex2 += (X - K) * (X - K)

        mean /= n

        # Here we use unbiased population estimate, to match sklearn StandardScaler
        # divide by n instead of (n - 1)
        variance = (ex2 - (ex * ex) / n) / (n - 1)
        std = np.sqrt(variance)

        self.fit_params = {"mean": mean, "std": std}
        print(f"transformer fit params {self.fit_params}")

    def transform(self, data_record):
        key = self.params["key"]
        mean = self.fit_params["mean"]
        std = self.fit_params["std"]

        # subtract by mean, divide by standar
        data_record[0][key] = (data_record[0][key] - mean) / std

        return data_record

    def postprocess(self, score):
        return score

    def save(self, path):
        io_utils.save_pickle(self.fit_params, "fit_params.pkl", path)

    def load(self, path):
        self.fit_params = io_utils.load_pickle("fit_params.pkl", path)


def add_input_noise(data_record, key, loc=0, scale=0.01):
    # Reminder data record is (X, y) or (X, y, w)
    # To add noise to an input, we need to index 0, key
    data_record[0][key] += np.random.normal(loc, scale, data_record[0][key].shape)

    return data_record


if __name__ == "__main__":
    vanilla_iris()
    overkill_iris()
