from typing import List

import tensorflow as tf

from barrage import config
from barrage.utils import import_utils


def build_network(cfg_model: dict, transform_params: dict) -> tf.keras.Model:
    """Build the network.

    Args:
        cfg_model: dict, model subsection of config.
        transform_params: dict, params from transformer.

    Returns:
        tf.keras.Model, network.

    Raises:
        TypeError, network not a tf.keras.Model.
    """
    path = cfg_model["network"]["import"]
    network_params = cfg_model["network"].get("params", {})

    net_func = import_utils.import_obj_with_search_modules(path)
    net = net_func(**network_params, **transform_params)

    if not isinstance(net, tf.keras.Model):
        raise TypeError(f"import network: {net} is not a tf.keras.Model")

    return net


def build_objective(cfg_model: dict) -> dict:
    """Build objective (loss, loss_weights, metrics, and sample_weight_mode)
    for each model output.

    Args:
        cfg_model: dict, model subsection of config.

    Returns:
        dict, objective
    """
    loss = {}
    loss_weights = {}
    sample_weight_mode = {}
    metrics = {}

    for output in cfg_model["outputs"]:
        name = output["name"]
        loss[name] = import_utils.import_loss(output["loss"])
        loss_weights[name] = output.get("loss_weight", 1.0)
        sample_weight_mode[name] = output.get("sample_weight_mode")
        metrics[name] = [
            import_utils.import_metric(m) for m in output.get("metrics", [])
        ]

    return {
        "loss": loss,
        "loss_weights": loss_weights,
        "sample_weight_mode": sample_weight_mode,
        "metrics": metrics,
    }


def check_output_names(cfg_model: dict, net: tf.keras.Model):
    """Check the net outputs in the config match the actual net.

    Args:
        cfg_model: dict, model subsection of config.
        net: tf.keras.Model, net.

    Raises:
        ValueError, mismatch between config and net.
    """
    config_net_outputs = {o["name"] for o in cfg_model["outputs"]}
    actual_net_outputs = set(net.output_names)
    if config_net_outputs != actual_net_outputs:
        raise ValueError(
            f"'config.model.outputs.names': {config_net_outputs} "
            f"mismatch actual model outputs: {actual_net_outputs} - "
            "order and names must exactly match"
        )


def sequential_from_config(layers: List[dict], **kwargs) -> tf.keras.Model:
    """Build a sequential model from a list of layer specifications.
    Supports references to network_params computed inside Transformers by specifying
    {{variable name}}.

    Args:
        layers: list[dict], layer imports.

    Returns:
        tf.keras.Model, network.
    """
    layers = config._render_params(layers, kwargs)
    network = tf.keras.models.Sequential()
    for layer in layers:

        if "import" not in layer:
            raise KeyError(f"layer {layer} missing 'import' key")
        if not layer.keys() <= {"import", "params"}:
            unexpected_keys = set(layer.keys()).difference({"import", "params"})
            raise KeyError(f"layer {layer} unexpected key(s): {unexpected_keys}")

        layer_cls = import_utils.import_obj_with_search_modules(
            layer["import"], search_modules=["tensorflow.keras.layers"]
        )
        layer_params = layer.get("params", {})
        network.add(layer_cls(**layer_params))

    return network
