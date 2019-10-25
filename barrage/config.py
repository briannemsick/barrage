import copy
import os

import jsonschema

from barrage import defaults as d
from barrage.utils import io_utils


def prepare_config(cfg: dict) -> dict:
    """Prepare config for use - apply defaults, validate schema.

    Args:
        cfg: dict, config.

    Returns:
        dict, validated config with defaults.
    """
    cfg = _merge_defaults(cfg)
    _validate_schema(cfg)
    return cfg


def _merge_defaults(cfg: dict) -> dict:
    """Merge default params to config.

    Args:
        cfg: dict, config.

    Returns:
        dict, config with defaults.

    Raises:
        jsonschema.ValidationError: invalid config params.
    """
    cfg["dataset"] = cfg.get("dataset", {})
    if not isinstance(cfg["dataset"], dict):
        raise jsonschema.ValidationError("config param 'dataset' must be a dict")
    cfg["dataset"]["transformer"] = cfg["dataset"].get("transformer", d.TRANSFORMER)
    cfg["dataset"]["augmentor"] = cfg["dataset"].get("augmentor", d.AUGMENTOR)

    cfg["solver"] = cfg.get("solver", {})
    if not isinstance(cfg["solver"], dict):
        raise jsonschema.ValidationError("config param 'solver' must be a dict")

    cfg["solver"]["batch_size"] = cfg["solver"].get("batch_size", d.BATCH_SIZE)
    cfg["solver"]["epochs"] = cfg["solver"].get("epochs", d.EPOCHS)
    cfg["solver"]["optimizer"] = cfg["solver"].get("optimizer", d.OPTIMIZER)

    cfg["services"] = cfg.get("services", {})
    if not isinstance(cfg["services"], dict):
        raise jsonschema.ValidationError("config param 'services' must be a dict")

    cfg["services"]["best_checkpoint"] = cfg["services"].get(
        "best_checkpoint", d.BEST_CHECKPOINT
    )
    cfg["services"]["tensorboard"] = cfg["services"].get("tensorboard", d.TENSORBOARD)
    cfg["services"]["train_early_stopping"] = cfg["services"].get(
        "train_early_stopping", d.TRAIN_EARLY_STOPPING
    )
    cfg["services"]["validation_early_stopping"] = cfg["services"].get(
        "validation_early_stopping", d.VALIDATION_EARLY_STOPPING
    )

    return copy.deepcopy(cfg)


def _validate_schema(cfg: dict):
    """Validate config params against schema.

    Args:
        cfg: dict, config.

    Raises:
        jsonschema.ValidationError: invalid config params.
    """
    schema = io_utils.load_json(
        "schema.json", os.path.abspath(os.path.dirname(__file__))
    )
    try:
        jsonschema.validate(cfg, schema)
    except jsonschema.ValidationError as err:
        raise jsonschema.ValidationError(f"invalid barrage config: {err}")

    # Check model outputs have unique names
    num_outputs = len(cfg["model"]["outputs"])
    num_unique_names = len({o["name"] for o in cfg["model"]["outputs"]})
    if num_outputs != num_unique_names:
        raise jsonschema.ValidationError(
            f"invalid barrage config: 'outputs' names are not unique"
        )

    # Check that multi-output networks have loss weights for each output
    if num_outputs > 1:
        if not all("loss_weight" in output for output in cfg["model"]["outputs"]):
            raise jsonschema.ValidationError(
                "invalid barrage config: each output in 'outputs' requires a "
                "'loss_weight' for multi output networks"
            )


def _render_params(cfg: dict, params: dict) -> dict:  # noqa C:901
    """Render a config or config section with params jinja style.

    Args:
        cfg: dict, config.
        params: dict, render params.

    Returns:
        dict, rendered config.
    """

    def _replace_item(obj, old, new):

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (list, dict)):
                    obj[k] = _replace_item(v, old, new)
                elif isinstance(v, str) and obj[k] == old:
                    obj[k] = new
        elif isinstance(obj, list):
            for k, v in enumerate(obj):
                if isinstance(v, (list, dict)):
                    obj[k] = _replace_item(v, old, new)
                elif isinstance(v, str) and obj[k] == old:
                    obj[k] = new

        return obj

    for k, v in params.items():
        # jinja style
        jk = "{{" + k + "}}"
        cfg = _replace_item(cfg, jk, v)

    return cfg
