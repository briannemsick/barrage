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
    metrics_names = ["loss"]
    result = services.create_all_services(artifact_dir, cfg_services, metrics_names)
    assert isinstance(result[0], tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(result[1], tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(result[2], tf.keras.callbacks.TensorBoard)
    assert isinstance(result[3], tf.keras.callbacks.CSVLogger)
    assert isinstance(result[4], tf.keras.callbacks.EarlyStopping)
    assert isinstance(result[5], tf.keras.callbacks.EarlyStopping)
    assert isinstance(result[6], tf.keras.callbacks.TerminateOnNaN)


@pytest.mark.parametrize(
    "cfg_services,metrics_names",
    [
        ({"best_checkpoint": {"monitor": "acc", "mode": "max"}}, ["loss", "acc"]),
        ({"best_checkpoint": {"monitor": "val_acc", "mode": "max"}}, ["loss", "acc"]),
    ],
)
def test_create_best_checkpoint(artifact_dir, cfg_services, metrics_names):
    result = services._create_best_checkpoint(artifact_dir, cfg_services, metrics_names)
    filepath = os.path.join(artifact_dir, services.BEST_CHECKPOINT, services.BEST_MODEL)
    expected = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_acc",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    assert isinstance(result, tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(expected, tf.keras.callbacks.ModelCheckpoint)
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
    assert isinstance(result, tf.keras.callbacks.ModelCheckpoint)
    assert isinstance(expected, tf.keras.callbacks.ModelCheckpoint)
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
    assert isinstance(result, tf.keras.callbacks.TensorBoard)
    assert isinstance(expected, tf.keras.callbacks.TensorBoard)
    assert vars(result) == vars(expected)


def test_csv_logger(artifact_dir):
    result = services._create_csv_logger(artifact_dir)
    filename = os.path.join(artifact_dir, services.CSV_LOGGER_FILENAME)
    expected = tf.keras.callbacks.CSVLogger(
        filename=filename, separator=",", append=True
    )
    assert isinstance(result, tf.keras.callbacks.CSVLogger)
    assert isinstance(expected, tf.keras.callbacks.CSVLogger)
    assert vars(result) == vars(expected)


@pytest.mark.parametrize(
    "cfg_services,metrics_names",
    [
        (
            {
                "train_early_stopping": {
                    "monitor": "acc",
                    "mode": "max",
                    "patience": 5,
                    "min_delta": 0.1,
                }
            },
            ["loss", "acc"],
        ),
        (
            {
                "train_early_stopping": {
                    "monitor": "val_acc",
                    "mode": "max",
                    "patience": 5,
                    "min_delta": 0.1,
                }
            },
            ["loss", "acc"],
        ),
    ],
)
def test_train_early_stopping(artifact_dir, cfg_services, metrics_names):
    result = services._create_train_early_stopping(cfg_services, metrics_names)
    expected = tf.keras.callbacks.EarlyStopping(
        monitor="acc", mode="max", min_delta=0.1, patience=5
    )
    assert isinstance(result, tf.keras.callbacks.EarlyStopping)
    assert isinstance(expected, tf.keras.callbacks.EarlyStopping)
    assert vars(result) == vars(expected)


@pytest.mark.parametrize(
    "cfg_services,metrics_names",
    [
        (
            {
                "validation_early_stopping": {
                    "monitor": "acc",
                    "mode": "max",
                    "patience": 5,
                    "min_delta": 0.1,
                }
            },
            ["loss", "acc"],
        ),
        (
            {
                "validation_early_stopping": {
                    "monitor": "val_acc",
                    "mode": "max",
                    "patience": 5,
                    "min_delta": 0.1,
                }
            },
            ["loss", "acc"],
        ),
    ],
)
def test_validation_early_stopping(artifact_dir, cfg_services, metrics_names):
    result = services._create_validation_early_stopping(cfg_services, metrics_names)
    expected = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", min_delta=0.1, patience=5
    )
    assert isinstance(result, tf.keras.callbacks.EarlyStopping)
    assert isinstance(expected, tf.keras.callbacks.EarlyStopping)
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


@pytest.mark.parametrize(
    "monitor,metrics_names,to_val,result",
    [
        ("loss", ["loss", "acc"], True, "val_loss"),
        ("acc", ["loss", "acc"], True, "val_acc"),
        ("val_loss", ["loss", "acc"], True, "val_loss"),
        ("val_acc", ["loss", "acc"], True, "val_acc"),
        ("auc", ["loss", "acc"], True, ValueError),
        ("val_auc", ["loss", "acc"], True, ValueError),
        ("loss", ["loss", "acc"], False, "loss"),
        ("acc", ["loss", "acc"], False, "acc"),
        ("val_loss", ["loss", "acc"], False, "loss"),
        ("val_acc", ["loss", "acc"], False, "acc"),
        ("auc", ["loss", "acc"], False, ValueError),
        ("val_auc", ["loss", "acc"], False, ValueError),
    ],
)
def test_force_monitor_to_mode(monitor, metrics_names, to_val, result):
    service_name = "unit-test"
    if isinstance(result, str):
        out = services._force_monitor_to_mode(
            monitor, metrics_names, to_val, service_name
        )
        assert out == result
    else:
        with pytest.raises(result):
            out = services._force_monitor_to_mode(
                monitor, metrics_names, to_val, service_name
            )
