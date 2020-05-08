import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from barrage import engine

NUM_SAMPLES = 100


@pytest.fixture
def cfg():
    return {
        "dataset": {
            "loader": {
                "import": "KeySelector",
                "params": {
                    "inputs": {"input": ["x1", "x2"]},
                    "outputs": {"output": ["y1", "y2"]},
                },
            },
            "transformer": {
                "import": "tests.unit.dataset.test_dataset.SimpleTransformer",
                "params": {"input_key": "input", "output_key": "output"},
            },
        },
        "model": {
            "network": {
                "import": "barrage.model.sequential_from_config",
                "params": {
                    "layers": [
                        {"import": "Input", "params": {"shape": 2, "name": "input"}},
                        {
                            "import": "Dense",
                            "params": {"units": 25, "activation": "relu"},
                        },
                        {
                            "import": "Dense",
                            "params": {
                                "units": 2,
                                "name": "output",
                                "activation": "linear",
                            },
                        },
                    ]
                },
            },
            "outputs": [
                {
                    "name": "output",
                    "loss": {"import": "mse"},
                    "metrics": [{"import": "mae"}],
                }
            ],
        },
        "solver": {
            "batch_size": 16,
            "epochs": 2,
            "optimizer": {
                "import": "Adam",
                "learning_rate": {
                    "import": "ExponentialDecay",
                    "params": {
                        "initial_learning_rate": 0.1,
                        "decay_steps": 100,
                        "decay_rate": 0.99,
                    },
                },
            },
        },
        "services": {
            "best_checkpoint": {"monitor": "val_loss", "mode": "min"},
            "tensorboard": {},
            "train_early_stopping": {
                "monitor": "loss",
                "mode": "min",
                "patience": 2,
                "min_delta": 1e-2,
                "verbose": 0,
            },
            "validation_early_stopping": {
                "monitor": "val_loss",
                "mode": "min",
                "patience": 2,
                "min_delta": 1e-2,
            },
        },
    }


def gen_records(num_samples):
    y = np.random.rand(num_samples, 2) + 100
    x = y + 50
    return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y1": y[:, 0], "y2": y[:, 1]})


def test_train(shared_artifact_dir, cfg):
    records_train = gen_records(NUM_SAMPLES)
    records_validation = gen_records(NUM_SAMPLES)

    bm = engine.BarrageModel(shared_artifact_dir)
    net = bm.train(cfg, records_train, records_validation)
    assert isinstance(net, tf.keras.models.Model)

    assert os.path.isdir(shared_artifact_dir)
    assert os.path.isdir(os.path.join(shared_artifact_dir, "dataset"))
    assert os.path.isdir(os.path.join(shared_artifact_dir, "best_checkpoint"))
    assert os.path.isdir(os.path.join(shared_artifact_dir, "resume_checkpoints"))
    assert os.path.isdir(os.path.join(shared_artifact_dir, "TensorBoard"))
    assert os.path.isfile(os.path.join(shared_artifact_dir, "training_report.csv"))
    assert os.path.isfile(os.path.join(shared_artifact_dir, "config.json"))
    assert os.path.isfile(os.path.join(shared_artifact_dir, "config.pkl"))
    assert os.path.isfile(os.path.join(shared_artifact_dir, "network_params.json"))
    assert os.path.isfile(os.path.join(shared_artifact_dir, "network_params.pkl"))

    assert os.path.isfile(
        os.path.join(shared_artifact_dir, "best_checkpoint", "model_best.ckpt.index")
    )
    assert os.path.isfile(
        os.path.join(
            shared_artifact_dir, "resume_checkpoints", "model_epoch_0001.ckpt.index"
        )
    )
    assert os.path.isfile(
        os.path.join(
            shared_artifact_dir, "resume_checkpoints", "model_epoch_0002.ckpt.index"
        )
    )


def test_predict(shared_artifact_dir):
    records_score = gen_records(NUM_SAMPLES)

    bm = engine.BarrageModel(shared_artifact_dir)
    bm.load()
    scores = bm.predict(records_score)

    assert len(scores) == NUM_SAMPLES

    for s in scores:
        assert list(s.keys()) == ["output"]
        assert len(s["output"]) == 2
        assert 98 <= s["output"][0] <= 102
        assert 98 <= s["output"][1] <= 102
