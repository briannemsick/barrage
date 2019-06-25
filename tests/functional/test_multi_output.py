import os

import numpy as np
import pandas as pd
import pytest
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model

from barrage import BarrageModel
from barrage.utils import io_utils

NUM_SAMPLES_TRAIN = 613
NUM_SAMPLES_VALIDATION = 216
NUM_SAMPLES_SCORE = 297


@pytest.fixture
def records_train():
    # Classification output and weight
    y_cls = np.random.randint(0, 3, NUM_SAMPLES_TRAIN).astype(np.float32)
    w_cls = y_cls * 0.1 + 1

    # x = input 1
    x1 = np.random.normal(0, 2.0, NUM_SAMPLES_TRAIN) + y_cls
    x2 = np.random.normal(-1.0, 0.25, NUM_SAMPLES_TRAIN) + y_cls
    x3 = np.random.normal(1.0, 0.1, NUM_SAMPLES_TRAIN) + y_cls

    # z = input 2
    z1 = np.random.normal(0.5, 0.25, NUM_SAMPLES_TRAIN) + y_cls
    z2 = np.random.randint(-1, 2, NUM_SAMPLES_TRAIN).astype(np.float32)

    # Regression output and temporal weights
    y_reg_1 = (
        -0.2 * x1 + 0.3 * x2 + 0.4 * x3 + np.random.normal(0, 0.01, NUM_SAMPLES_TRAIN)
    )
    y_reg_2 = -0.5 * x3 + 0.5 * z1 * z2 + np.random.normal(0, 0.01, NUM_SAMPLES_TRAIN)
    w_reg = np.maximum(y_reg_1, 2)

    # Sample
    s = np.random.randint(1, 5, NUM_SAMPLES_TRAIN).astype(np.float32)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "z1": z1,
            "z2": z2,
            "y_cls": y_cls,
            "y_reg_1": y_reg_1,
            "y_reg_2": y_reg_2,
            "w_cls": w_cls,
            "w_reg": w_reg,
            "sample": s,
        }
    )
    return df


@pytest.fixture
def records_validation():
    # Classification output and weight
    y_cls = np.random.randint(0, 3, NUM_SAMPLES_VALIDATION).astype(np.float32)
    w_cls = y_cls * 0.1 + 1

    # x = input 1
    x1 = np.random.normal(0, 2.0, NUM_SAMPLES_VALIDATION) + y_cls
    x2 = np.random.normal(-1.0, 0.25, NUM_SAMPLES_VALIDATION) + y_cls
    x3 = np.random.normal(1.0, 0.1, NUM_SAMPLES_VALIDATION) + y_cls

    # z = input 2
    z1 = np.random.normal(0.5, 0.25, NUM_SAMPLES_VALIDATION) + y_cls
    z2 = np.random.randint(-1, 2, NUM_SAMPLES_VALIDATION).astype(np.float32)

    # Regression output and temporal weights
    y_reg_1 = (
        -0.2 * x1
        + 0.3 * x2
        + 0.4 * x3
        + np.random.normal(0, 0.01, NUM_SAMPLES_VALIDATION)
    )
    y_reg_2 = (
        -0.5 * x3 + 0.5 * z1 * z2 + np.random.normal(0, 0.01, NUM_SAMPLES_VALIDATION)
    )
    w_reg = np.maximum(y_reg_1, 2)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "z1": z1,
            "z2": z2,
            "y_cls": y_cls,
            "y_reg_1": y_reg_1,
            "y_reg_2": y_reg_2,
            "w_cls": w_cls,
            "w_reg": w_reg,
        }
    )
    return df


@pytest.fixture
def records_score():
    # Classification output
    y_cls = np.random.randint(0, 3, NUM_SAMPLES_SCORE).astype(np.float32)

    # x = input 1
    x1 = np.random.normal(0, 2.0, NUM_SAMPLES_SCORE) + y_cls
    x2 = np.random.normal(-1.0, 0.25, NUM_SAMPLES_SCORE) + y_cls
    x3 = np.random.normal(1.0, 0.1, NUM_SAMPLES_SCORE) + y_cls

    # z = input 2
    z1 = np.random.normal(0.5, 0.25, NUM_SAMPLES_SCORE) + y_cls
    z2 = np.random.randint(-1, 2, NUM_SAMPLES_SCORE).astype(np.float32)

    # Regression output
    y_reg_1 = (
        -0.2 * x1 + 0.3 * x2 + 0.4 * x3 + np.random.normal(0, 0.01, NUM_SAMPLES_SCORE)
    )
    y_reg_2 = -0.5 * x3 + 0.5 * z1 * z2 + np.random.normal(0, 0.01, NUM_SAMPLES_SCORE)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "z1": z1,
            "z2": z2,
            "y_cls": y_cls,
            "y_reg_1": y_reg_1,
            "y_reg_2": y_reg_2,
        }
    )
    return df


def add_noise(data_record, ind, input_key, scale=0.01):
    data_record[ind][input_key] += np.random.normal(
        0, scale, data_record[ind][input_key].shape
    )
    return data_record


def net():
    input_x = layers.Input(shape=(3,), name="input_x")
    input_z = layers.Input(shape=(2,), name="input_z")

    dense_1_x = layers.Dense(10, activation="relu")(input_x)
    dense_1_z = layers.Dense(10, activation="relu")(input_z)
    add = layers.Add()([dense_1_x, dense_1_z])
    dense_2 = layers.Dense(10, activation="relu")(add)
    dense_3 = layers.Dense(10, activation="relu")(dense_2)

    output_cls = layers.Dense(3, activation="softmax", name="classification")(dense_3)
    output_reg = layers.Dense(2, activation="linear", name="regression")(dense_3)

    return Model(inputs=[input_x, input_z], outputs=[output_reg, output_cls])


def test_multi_output(artifact_dir, records_train, records_validation, records_score):
    loc = os.path.abspath(os.path.dirname(__file__))
    cfg = io_utils.load_json("config_multi_output.json", loc)

    bm = BarrageModel(artifact_dir)
    bm.train(cfg, records_train, records_validation)
    scores = bm.predict(records_score)

    classification = [np.argmax(score["classification"]) for score in scores]
    regression_1 = [score["regression"][0] for score in scores]
    regression_2 = [score["regression"][1] for score in scores]

    df_scores = pd.DataFrame(
        {
            "classification": classification,
            "regression_1": regression_1,
            "regression_2": regression_2,
        }
    )

    assert (df_scores["classification"] == records_score["y_cls"]).mean() > 0.5
    assert abs((df_scores["regression_1"] - records_score["y_reg_1"]).mean()) < 0.5
    assert abs((df_scores["regression_2"] - records_score["y_reg_2"]).mean()) < 0.5
