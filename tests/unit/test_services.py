import os

import pytest
import tensorflow as tf

from barrage import services


def test_make_artifact_dir(artifact_dir):
    services.make_artifact_dir(artifact_dir)
    assert os.path.isdir(artifact_dir)
    assert os.path.isdir(os.path.join(artifact_dir, services.BEST_CHECKPOINT))
    assert os.path.isdir(os.path.join(artifact_dir, services.RESUME_CHECKPOINTS))
    assert os.path.isdir(os.path.join(artifact_dir, services.TENSORBOARD))
    assert os.path.isdir(os.path.join(artifact_dir, services.DATASET))

    # 2nd creation
    with pytest.raises(OSError):
        services.make_artifact_dir(artifact_dir)


def test_create_all_services(artifact_dir):
    cfg_services = {
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
    }
    result = services.create_all_services(artifact_dir, cfg_services)
    assert isinstance(result[0], tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(result[1], tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(result[2], tf.keras.callbacks.TensorBoard)
    assert isinstance(result[3], tf.keras.callbacks.CSVLogger)
    assert isinstance(result[4], tf.keras.callbacks.EarlyStopping)
    assert isinstance(result[5], tf.keras.callbacks.EarlyStopping)
    assert isinstance(result[6], tf.keras.callbacks.TerminateOnNaN)


@pytest.mark.parametrize(
    "cfg_services",
    [
        {"best_checkpoint": {"monitor": "acc", "mode": "max"}},
        {"best_checkpoint": {"monitor": "val_acc", "mode": "max"}},
    ],
)
def test_create_best_checkpoint(artifact_dir, cfg_services):
    result = services._create_best_checkpoint(artifact_dir, cfg_services)
    filepath = os.path.join(artifact_dir, services.BEST_CHECKPOINT, services.BEST_MODEL)
    expected = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_acc",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    assert type(result) == type(expected)
    assert vars(result) == vars(expected)


def test_create_resume_checkpoint(artifact_dir):
    result = services._create_resume_checkpoint(artifact_dir)
    filepath = os.path.join(
        artifact_dir, services.RESUME_CHECKPOINTS, services.RESUME_MODEL
    )
    expected = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        mode="min",
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
    )
    assert type(result) == type(expected)
    assert vars(result) == vars(expected)


@pytest.mark.parametrize(
    "cfg_services",
    [
        {"tensorboard": {"write_grads": True}},
        {"tensorboard": {"log_dir": "unit-test", "write_grads": True}},
    ],
)
def test_create_tensorboard(artifact_dir, cfg_services):
    result = services._create_tensorboard(artifact_dir, cfg_services)
    log_dir = os.path.join(artifact_dir, services.TENSORBOARD)
    expected = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True)
    assert type(result) == type(expected)
    assert vars(result) == vars(expected)


def test_csv_logger(artifact_dir):
    result = services._create_csv_logger(artifact_dir)
    filename = os.path.join(artifact_dir, services.CSV_LOGGER_FILENAME)
    expected = tf.keras.callbacks.CSVLogger(
        filename=filename, separator=",", append=True
    )
    assert type(result) == type(expected)
    assert vars(result) == vars(expected)


@pytest.mark.parametrize(
    "cfg_services",
    [
        {
            "train_early_stopping": {
                "monitor": "acc",
                "mode": "max",
                "patience": 5,
                "min_delta": 0.1,
            }
        },
        {
            "train_early_stopping": {
                "monitor": "val_acc",
                "mode": "max",
                "patience": 5,
                "min_delta": 0.1,
            }
        },
    ],
)
def test_train_early_stopping(artifact_dir, cfg_services):
    result = services._create_train_early_stopping(cfg_services)
    expected = tf.keras.callbacks.EarlyStopping(
        monitor="acc", mode="max", min_delta=0.1, patience=5
    )
    assert type(result) == type(expected)
    assert vars(result) == vars(expected)


@pytest.mark.parametrize(
    "cfg_services",
    [
        {
            "validation_early_stopping": {
                "monitor": "acc",
                "mode": "max",
                "patience": 5,
                "min_delta": 0.1,
            }
        },
        {
            "validation_early_stopping": {
                "monitor": "val_acc",
                "mode": "max",
                "patience": 5,
                "min_delta": 0.1,
            }
        },
    ],
)
def test_validation_early_stopping(artifact_dir, cfg_services):
    result = services._create_validation_early_stopping(cfg_services)
    expected = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", min_delta=0.1, patience=5
    )
    assert type(result) == type(expected)
    assert vars(result) == vars(expected)


def test_get_best_checkpoint_filepath(artifact_path):
    result = services.get_best_checkpoint_filepath(artifact_path)
    expected = os.path.join(
        artifact_path, services.BEST_CHECKPOINT, services.BEST_MODEL
    )
    assert result == expected


def test_get_resume_checkpoints_filepath(artifact_path):
    result = services.get_resume_checkpoints_filepath(artifact_path)
    expected = os.path.join(
        artifact_path, services.RESUME_CHECKPOINTS, services.RESUME_MODEL
    )
    assert result == expected
