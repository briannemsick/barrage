from typing import List

from tensorflow.python.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

from barrage.utils import import_utils


def build_optimizer(cfg_solver: dict) -> optimizer_v2.OptimizerV2:
    """Build the optimizer.

    Args:
        cfg_solver: dict, solver subsection of config.

    Returns:
        optimizer_v2.OptimizerV2, tf.keras v2 optimizer.

    Raises:
        TypeError, optimizer not an OptimizerV2.
    """
    path = cfg_solver["optimizer"]["import"]
    params = cfg_solver["optimizer"].get("params", {})
    learning_rate = cfg_solver["optimizer"]["learning_rate"]

    opt_cls = import_utils.import_obj_with_search_modules(
        path, ["tensorflow.keras.optimizers"], True
    )
    opt = opt_cls(learning_rate=learning_rate, **params)

    if not isinstance(opt, optimizer_v2.OptimizerV2):
        raise TypeError(f"import optimizer: {opt} is not an OptimizerV2")

    return opt


def create_learning_rate_reducer(
    cfg_solver: dict, metrics_names: List[str]
) -> ReduceLROnPlateau:
    """Create a ReduceLROnPlateau callback.

    Args:
        cfg_solver: dict, solver subsection of config.
        metrics_names: list[str], 'metrics' names.

    Returns:
        ReduceLROnPlateau, ReduceLROnPlateau callback.

    Raises:
        ValueError, monitor not in 'metrics' names.
    """
    monitor = cfg_solver["learning_rate_reducer"]["monitor"]
    val_metrics_names = [f"val_{mm}" for mm in metrics_names]
    if (monitor not in metrics_names) and (monitor not in val_metrics_names):
        raise ValueError(
            f"monitor: {monitor} not found in model metrics names: "
            f"{metrics_names + val_metrics_names}"
        )
    params = cfg_solver["learning_rate_reducer"]
    params["verbose"] = 1
    return ReduceLROnPlateau(**params)


def create_learning_rate_scheduler(cfg_solver: dict) -> LearningRateScheduler:
    """Create a LearningRateScheduler callback.

    Args:
        cfg_solver: dict, solver subsection of config.

    Returns:
        LearningRateScheduler, LearningRateScheduler callback.
    """
    schedule = import_utils.import_partial_wrap_func(
        cfg_solver["learning_rate_scheduler"]
    )
    return LearningRateScheduler(schedule=schedule, verbose=1)
