from functools import partial
import importlib
import inspect
from typing import Callable, List, Union

from tensorflow.python.keras import losses, metrics

from barrage import logger


def import_obj_with_search_modules(
    python_path: str, search_modules: List[str] = None, search_both_cases=False
) -> Callable:
    """Import a object (variable, class, function, etc...) from a python path.
    search_modules are used to potentially shorten the python path by eliminating the
    module part of the string. Optionally check both uncapitalized and capitalized
    version of the object name.

    Args:
        python_path: str, full python path or object name (in conjunction with
             search_modules).
        search_modules: list[str], python modules against short path.
        search_both_cases: bool, try capitalized and uncapitalized object name.

    Returns:
        callable, imported object.

    Raises:
        ImportError, object not found.
    """
    python_path = _maybe_fix_tensorflow_python_path(python_path)
    try:
        module_path, obj_name = python_path.rsplit(".", 1)
    except ValueError:
        module_path = ""
        obj_name = python_path

    if module_path:
        module = importlib.import_module(module_path)
        if not hasattr(module, obj_name):
            raise ImportError(
                f"object: {obj_name} was not found in module {module_path}"
            )
        return getattr(module, obj_name)
    else:
        if search_modules is None:
            raise ImportError(f"object: {obj_name} not found, no module provided.")

        obj_names = _make_both_cases(obj_name) if search_both_cases else [obj_name]
        for search_module_path in search_modules:
            module = importlib.import_module(search_module_path)
            for name in obj_names:
                if hasattr(module, name):
                    return getattr(module, name)

        raise ImportError(
            f"object: {obj_name} was not found in the searched "
            f"modules {search_modules}"
        )


def import_or_alias(python_path: str) -> Union[Callable, str]:
    """Import an object from a full python path or assume it is an alias for some object
    in TensorFlow (e.g. "categorical_crossentropy").

    Args:
        python_path: str, full python path or alias.

    Returns:
        callable/str, import object/alias.

    Raises:
        ImportError, object not found.
    """
    python_path = _maybe_fix_tensorflow_python_path(python_path)
    try:
        module_path, obj_name = python_path.rsplit(".", 1)
    except ValueError:
        return python_path

    module = importlib.import_module(module_path)

    if not hasattr(module, obj_name):
        raise ImportError(f"object: {obj_name} was not found in module {module_path}")
    return getattr(module, obj_name)


def import_partial_wrap_func(import_block: dict) -> Callable:
    """Import a function from an import block and  wrap with partial.

    Args:
        import_block: dict, {"import": str, "params": dict}.

    Returns:
        function.
    """
    func = import_obj_with_search_modules(import_block["import"])
    params = import_block.get("params", {})
    return partial(func, **params)


def import_loss(import_block: dict) -> Union[losses.Loss, str]:
    """Import a loss from an import block.

    Args:
        import_block: dict, {"import": str, "params": dict}.

    Returns:
        Loss/str, non-alias loss or alias.

    Raises:
        TypeError, non-alias loss is not a Loss.
    """
    loss = import_or_alias(import_block["import"])
    if isinstance(loss, str):
        return loss
    else:
        params = import_block.get("params", {})
        if not inspect.isclass(loss):
            raise TypeError(
                f"import loss: {loss} must be a class for v2 tensorflow.keras API"
            )
        loss = loss(**params)
        if not isinstance(loss, losses.Loss):
            raise TypeError(
                f"import loss: {loss} is not a tensorflow.python.keras.losses.Loss"
            )
        return loss


def import_metric(import_block: dict) -> Union[metrics.Metric, losses.Loss, str]:
    """Import a metric from an import block.

    Note: A loss can be a metric, but a metric cannot always be a loss.

    Args:
        import_block: dict, {"import": str, "params": dict}.

    Returns:
        Metric/Loss/str, non-alias metric or alias.

    Raises:
        TypeError, non-alias metric is not a Metric/loss.
    """
    metric = import_or_alias(import_block["import"])
    if isinstance(metric, str):
        return metric
    else:
        params = import_block.get("params", {})
        if not inspect.isclass(metric):
            raise TypeError(
                f"import metric: {metric} must be a class "
                "for v2 tensorflow.keras API"
            )
        metric = metric(**params)
        if not isinstance(metric, (metrics.Metric, losses.Loss)):
            raise TypeError(
                f"import metric: {metric} is not a "
                "tensorflow.python.keras.metrics.Metric or "
                "tensorflow.python.keras.losses.Loss"
            )
        return metric


def _maybe_fix_tensorflow_python_path(python_path: str) -> str:
    """Fix a common mistake python path mistake tf.keras vs tensorflow.keras.

    Args:
        python_path: str, python path.

    Returns:
        str, python path.
    """
    if python_path.startswith("tf.keras"):
        python_path = python_path.replace("tf.keras", "tensorflow.keras")
        logger.warning(
            f"fixing import {python_path} - replacing tf.keras "
            "with tensorflow.python.keras"
        )
    return python_path


def _make_both_cases(s: str) -> List[str]:
    """Capitalize and uncapitalize first letter of string. Original case first.

    Args:
        s, str.

    Return:
        list[str], capitalized and uncapitalized version of string.
    """
    if s:
        upper = s[:1].upper() + s[1:]
        lower = s[:1].lower() + s[1:]
        if upper == s:
            return [upper, lower]
        else:
            return [lower, upper]
    else:
        return [""]
