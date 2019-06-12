import os

import numpy as np
import pandas as pd
import pytest
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model

from barrage import BarrageModel
from barrage.dataset import RecordMode, RecordTransformer
from barrage.utils import io_utils

NUM_SAMPLES_TRAIN = 407
NUM_SAMPLES_VALIDATION = 193
NUM_SAMPLES_SCORE = 122


@pytest.fixture
def records_train():
    y = np.random.randint(0, 3, NUM_SAMPLES_TRAIN).astype(np.float32)
    x1 = np.random.normal(0, 2.0, NUM_SAMPLES_TRAIN) + y
    x2 = np.random.normal(-1.0, 1.0, NUM_SAMPLES_TRAIN) + y
    x3 = np.random.normal(1.0, 0.5, NUM_SAMPLES_TRAIN) + y
    x4 = np.random.normal(0.5, 0.25, NUM_SAMPLES_TRAIN) + y
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    df["label"] = df["y"].map({0: "class_0", 1: "class_1", 2: "class_2"})
    return df


@pytest.fixture
def records_validation():
    y = np.random.randint(0, 3, NUM_SAMPLES_VALIDATION).astype(np.float32)
    x1 = np.random.normal(0, 2.0, NUM_SAMPLES_VALIDATION) + y
    x2 = np.random.normal(-1.0, 1.0, NUM_SAMPLES_VALIDATION) + y
    x3 = np.random.normal(1.0, 0.5, NUM_SAMPLES_VALIDATION) + y
    x4 = np.random.normal(0.5, 0.25, NUM_SAMPLES_VALIDATION) + y
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    df["label"] = df["y"].map({0: "class_0", 1: "class_1", 2: "class_2"})
    return df


@pytest.fixture
def records_score():
    y = np.random.randint(0, 3, NUM_SAMPLES_SCORE).astype(np.float32)
    x1 = np.random.normal(0, 2.0, NUM_SAMPLES_SCORE) + y
    x2 = np.random.normal(-1.0, 1.0, NUM_SAMPLES_SCORE) + y
    x3 = np.random.normal(1.0, 0.5, NUM_SAMPLES_SCORE) + y
    x4 = np.random.normal(0.5, 0.25, NUM_SAMPLES_SCORE) + y
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    df["label"] = df["y"].map({0: "class_0", 1: "class_1", 2: "class_2"})
    return df


class Transformer(RecordTransformer):
    def __init__(self, mode, loader, params):
        super().__init__(mode, loader, params)
        self.output_column = list(self.loader.params["outputs"].values())[0][0]
        self.output_name = list(self.loader.params["outputs"].keys())[0]

    def fit(self, records):
        class_names = records[self.output_column].unique()
        self.class_map = {ii: class_names[ii] for ii in range(len(class_names))}
        self.inverse_class_map = dict(map(reversed, self.class_map.items()))

        self.network_params = {
            "input_dim": len(list(self.loader.params["inputs"].values())[0]),
            "num_classes": len(class_names),
        }

    def transform(self, record):
        if self.mode == RecordMode.TRAIN or self.mode == RecordMode.VALIDATION:
            val = self.inverse_class_map[record[1][self.output_name][0]]
            record[1][self.output_name] = np.array(val)

        return record

    def postprocess(self, score):
        ind = np.argmax(score[self.output_name])
        score[self.output_name] = self.class_map[ind]
        return score

    def save(self, path):
        io_utils.save_pickle(self.class_map, "class_map.pkl", path)
        io_utils.save_pickle(self.inverse_class_map, "inverse_class_map.pkl", path)

    def load(self, path):
        self.class_map = io_utils.load_pickle("class_map.pkl", path)
        self.inverse_class_map = io_utils.load_pickle("inverse_class_map.pkl", path)


def net(input_dim, dense_dim, num_dense, num_classes):
    inputs = layers.Input(shape=(input_dim,), name="input")
    x = inputs
    for _ in range(num_dense):
        x = layers.Dense(dense_dim, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


def test_simple_output(artifact_dir, records_train, records_validation, records_score):
    loc = os.path.abspath(os.path.dirname(__file__))
    cfg = io_utils.load_json("config_single_output.json", loc)

    bm = BarrageModel(artifact_dir)
    bm.train(cfg, records_train, records_validation)
    scores = bm.predict(records_score)

    df_scores = pd.DataFrame(scores)
    assert (df_scores["softmax"] == records_score["label"]).mean() >= 0.90
