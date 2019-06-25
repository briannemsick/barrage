import tensorflow as tf

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
    params = cfg_model["network"].get("params", {})

    net_func = import_utils.import_obj_with_search_modules(path)
    net = net_func(**params, **transform_params)

    if not isinstance(net, tf.keras.Model):
        raise TypeError(f"import network: {net} is not a tf.keras.Model")

    return net


def build_objective(cfg_model: dict):
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
    config_net_outputs = set(o["name"] for o in cfg_model["outputs"])
    actual_net_outputs = set(net.output_names)
    if config_net_outputs != actual_net_outputs:
        raise ValueError(
            f"'config.model.outputs.names': {config_net_outputs} "
            f"mismatch actual model outputs: {actual_net_outputs} - "
            "order and names must exactly match"
        )
