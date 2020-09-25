import os
from typing import List

from tensorflow.python.keras import callbacks

from barrage import logger

DATASET = "dataset"
BEST_CHECKPOINT = "best_checkpoint"
BEST_MODEL = "model_best.ckpt"
RESUME_CHECKPOINTS = "resume_checkpoints"
RESUME_MODEL = "model_epoch_{epoch:04d}.ckpt"
TENSORBOARD = "TensorBoard"
CSV_LOGGER_FILENAME = "training_report.csv"
REQUIRED_SUBDIRS = [DATASET, BEST_CHECKPOINT, RESUME_CHECKPOINTS, TENSORBOARD]


def make_artifact_dir(artifact_dir: str):
    """Make the artifact directory and all required subdirectories.

    Args:
        artifact_dir: str, path to artifact directory.

    Raises:
        OSError, artifact_dir already exists.
    """
    if os.path.isdir(artifact_dir):
        raise OSError(f"artifact_dir: {artifact_dir} already exists")

    os.mkdir(artifact_dir)
    for subdir in REQUIRED_SUBDIRS:
        os.mkdir(os.path.join(artifact_dir, subdir))


def create_all_services(
    artifact_dir: str, cfg_services: dict
) -> List[callbacks.Callback]:
    """Create all services (callbacks).

    Args:
        artifact_dir: str, path to artifact directory.
        cfg_services: dict, services subsection of config.

    Returns:
        list[Callback], all services.
    """
    return [
        _create_best_checkpoint(artifact_dir, cfg_services),
        _create_resume_checkpoint(artifact_dir),
        _create_tensorboard(artifact_dir, cfg_services),
        _create_csv_logger(artifact_dir),
        _create_train_early_stopping(cfg_services),
        _create_validation_early_stopping(cfg_services),
        callbacks.TerminateOnNaN(),
    ]


def _create_best_checkpoint(
    artifact_dir: str, cfg_services: dict
) -> callbacks.ModelCheckpoint:
    """Create a callback that saves the best model.

    Args:
        artifact_dir: str, path to artifact directory.
        cfg_services: dict, services subsection of config.

    Returns:
        ModelCheckpoint, callback that saves the best model.
    """
    checkpoint_params = cfg_services["best_checkpoint"]
    checkpoint_params["monitor"] = _force_monitor_to_mode(
        checkpoint_params["monitor"], True, "best_checkpoint"
    )
    filepath = get_best_checkpoint_filepath(artifact_dir)
    return callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=checkpoint_params["monitor"],
        mode=checkpoint_params["mode"],
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )


def _create_resume_checkpoint(artifact_dir: str) -> callbacks.ModelCheckpoint:
    """Create a callback that saves the model every epoch.

    Args:
        artifact_dir: str, path to artifact directory.

    Returns:
        ModelCheckpoint, callback that saves the model every epoch.
    """
    filepath = get_resume_checkpoints_filepath(artifact_dir)
    return callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        mode="min",
        verbose=0,
        save_best_only=False,
        save_weights_only=True,
    )


def _create_tensorboard(artifact_dir: str, cfg_services: dict) -> callbacks.TensorBoard:
    """Create a TensorBoard callback.

    Args:
        artifact_dir: str, path to artifact directory.
        cfg_services: dict, services subsection of config.

    Returns:
        TensorBoard, Tensorboard callback.
    """
    tensorboard_params = cfg_services["tensorboard"]
    if "log_dir" in tensorboard_params:
        logger.warning("'log_dir' automatically handled for 'tensorboard' service")
    tensorboard_params["log_dir"] = os.path.join(artifact_dir, TENSORBOARD)
    return callbacks.TensorBoard(**tensorboard_params)


def _create_csv_logger(artifact_dir: str) -> callbacks.CSVLogger:
    """Create a CSVLogger callback.

    Args:
        artifact_dir: str, path to artifact directory.

    Returns:
        CSVLogger, CSVLogger callbackk.
    """
    filename = os.path.join(artifact_dir, CSV_LOGGER_FILENAME)
    return callbacks.CSVLogger(filename=filename, separator=",", append=True)


def _create_train_early_stopping(cfg_services: dict) -> callbacks.EarlyStopping:
    """Create an early stopping callback that monitors a training 'metric'.

    Args:
        cfg_services: dict, services subsection of config.

    Returns:
        EarlyStopping, EarlyStopping callback that monitors a training 'metric'.
    """
    early_stopping_params = cfg_services["train_early_stopping"]
    early_stopping_params["monitor"] = _force_monitor_to_mode(
        early_stopping_params["monitor"], False, "train_early_stopping"
    )
    return callbacks.EarlyStopping(**early_stopping_params)


def _create_validation_early_stopping(cfg_services: dict) -> callbacks.EarlyStopping:
    """Create an early stopping callback that monitors a validation 'metric'.

    Args:
        cfg_services: dict, services subsection of config.

    Returns:
        EarlyStopping, EarlyStopping callback that monitors a validation 'metric'.
    """
    early_stopping_params = cfg_services["validation_early_stopping"]
    early_stopping_params["monitor"] = _force_monitor_to_mode(
        early_stopping_params["monitor"], True, "validation_early_stopping"
    )
    return callbacks.EarlyStopping(**early_stopping_params)


def get_best_checkpoint_filepath(artifact_dir: str) -> str:
    """Get the filepath for the best checkpoint.

    Args:
        artifact_dir: str, path to artifact directory.

    Returns:
        str, filepath for best checkpoint directory.
    """
    return os.path.join(artifact_dir, BEST_CHECKPOINT, BEST_MODEL)


def get_resume_checkpoints_filepath(artifact_dir: str) -> str:
    """Get the filepath for the resume checkpoints.

    Args:
        artifact_dir: str, path to artifact directory.

    Returns:
        str, filepath for resume checkpoints.
    """
    return os.path.join(artifact_dir, RESUME_CHECKPOINTS, RESUME_MODEL)


def _force_monitor_to_mode(monitor: str, to_val: bool, service_name: str) -> str:
    """Force a monitor quantity to either train or validation mode. For
    example 'loss' - train, 'val_loss' - validation.

    Args:
        monitor: str, metric to monitor.
        to_val: bool, validation if true, else false.
        service_name: str, corresponding service (for warning purposes).

    Returns:
        str, monitor maybe forced.
    """
    if to_val and not monitor.startswith("val_"):
        monitor = f"val_{monitor}"
        logger.warning(
            f"corrected 'monitor' to validation verison: {monitor} "
            f"for service: {service_name}"
        )
    elif not to_val and monitor.startswith("val_"):
        monitor = monitor[4:]
        logger.warning(
            f"corrected 'monitor' to train verison: {monitor} "
            f"for service: {service_name}"
        )
    return monitor
