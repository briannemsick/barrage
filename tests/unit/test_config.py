import jsonschema
import pytest

from barrage.config import config
from barrage.config import defaults as d


def test_merge_defaults_dataset():
    cfg = {}
    result = config._merge_defaults(cfg.copy())

    assert isinstance(result, dict)
    assert result["dataset"]["transformer"] == d.TRANSFORMER
    assert result["dataset"]["augmentor"] == d.AUGMENTOR

    cfg = {
        "dataset": {
            "transformer": {"import": "IdentityTransformer"},
            "augmentor": {"import": "unit-test", "params": {"unit": "test"}},
        }
    }
    result = config._merge_defaults(cfg.copy())
    assert result["dataset"] == cfg["dataset"]

    cfg = {"dataset": []}
    with pytest.raises(jsonschema.ValidationError):
        config._merge_defaults(cfg)


def test_merge_defaults_solver():
    cfg = {}
    result = config._merge_defaults(cfg.copy())

    assert isinstance(result, dict)
    assert result["solver"]["batch_size"] == d.BATCH_SIZE
    assert result["solver"]["epochs"] == d.EPOCHS
    assert result["solver"]["optimizer"] == d.OPTIMIZER

    cfg = {"solver": {"batch_size": 42, "epochs": 7, "optimizer": {"import": "SGD"}}}
    result = config._merge_defaults(cfg.copy())
    assert result["solver"] == cfg["solver"]

    cfg = {"solver": []}
    with pytest.raises(jsonschema.ValidationError):
        config._merge_defaults(cfg)


def test_merge_defaults_services():
    cfg = {}
    result = config._merge_defaults(cfg.copy())

    assert isinstance(result, dict)
    assert result["services"]["best_checkpoint"] == d.BEST_CHECKPOINT
    assert result["services"]["tensorboard"] == d.TENSORBOARD
    assert result["services"]["train_early_stopping"] == d.TRAIN_EARLY_STOPPING
    assert (
        result["services"]["validation_early_stopping"] == d.VALIDATION_EARLY_STOPPING
    )

    cfg = {
        "services": {
            "best_checkpoint": {"monitor": "val_unit_test", "mode": "min"},
            "tensorboard": {"batch_size": 42},
            "train_early_stopping": {
                "monitor": "unit_test",
                "mode": "min",
                "min_delta": 1e-7,
                "patience": 42,
            },
            "validation_early_stopping": {
                "monitor": "val_unit_test",
                "mode": "max",
                "min_delta": 1e-7,
                "patience": 42,
            },
        }
    }
    result = config._merge_defaults(cfg.copy())
    assert result["services"] == cfg["services"]

    cfg = {"services": []}
    with pytest.raises(jsonschema.ValidationError):
        config._merge_defaults(cfg)


@pytest.fixture
def base_cfg():
    return {
        "dataset": {
            "loader": {"import": "my_loader", "params": {"unit": "test"}},
            "transformer": {"import": "my_tr", "params": {"unit": "test"}},
            "augmentor": [
                {"import": "my_aug_1", "params": {"unit": "test"}},
                {"import": "my_aug_2"},
            ],
        },
        "model": {
            "network": {"import": "my_net", "params": {"unit": "test"}},
            "outputs": [
                {
                    "name": "classification",
                    "loss": {"import": "crossentropy", "params": {"name": "ce"}},
                    "loss_weight": 1,
                },
                {
                    "name": "regression",
                    "loss": {"import": "mse"},
                    "loss_weight": 1,
                    "sample_weight_mode": "temporal",
                    "metrics": [{"import": "mse"}, {"import": "mae"}],
                },
            ],
        },
        "solver": {
            "optimizer": {
                "import": "Adam",
                "learning_rate": 1e-3,
                "params": {"beta1": 0.9},
            },
            "learning_rate_reducer": {
                "monitor": "val_loss",
                "mode": "min",
                "patience": 5,
                "factor": 0.07,
            },
            "batch_size": 32,
            "epochs": 10,
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


def _test_validate_schema_pass_fail(cfg, result):
    """Helper function to test if config should pass or fail."""
    if result:
        config._validate_schema(cfg)
    else:
        with pytest.raises(jsonschema.ValidationError):
            config._validate_schema(cfg)


def test_base_cfg(base_cfg):
    cfg_1 = base_cfg.copy()
    _test_validate_schema_pass_fail(base_cfg, True)
    cfg_2 = base_cfg.copy()
    assert cfg_1 == cfg_2


@pytest.mark.parametrize("section", ["dataset", "model", "solver", "services"])
def test_validate_schema_missing_sections(base_cfg, section):
    del base_cfg[section]
    _test_validate_schema_pass_fail(base_cfg, False)


def test_validate_schema_extra_sections(base_cfg):
    base_cfg["unit"] = "test"
    _test_validate_schema_pass_fail(base_cfg, False)


def set_nested(d, lst, val):

    from functools import reduce
    import operator

    loc = reduce(operator.getitem, lst[:-1], d)

    if val is not None:
        loc[lst[-1]] = val
    else:
        del loc[lst[-1]]

    return d


@pytest.mark.parametrize(
    "lst,val,result",
    [
        (["loader"], "my_loader", False),
        (["loader"], None, False),
        (["loader", "params"], None, True),
        (["loader", "import"], None, False),
        (["loader"], {"import": "my_loader", "params": {"unit": "test"}}, True),
        (["transformer"], "my_tr", False),
        (["transformer"], None, False),
        (["transformer", "params"], None, True),
        (["transformer", "import"], None, False),
        (["transformer"], {"import": "my_tr", "params": {"unit": "test"}}, True),
        (["augmentor"], [], True),
        (["augmentor"], False, False),
        (["augmentor"], [{"import": "my_aug"}], True),
        (["augmentor"], [{"import": "my_aug1"}, {"import": "my_aug2"}], True),
        (["augmentor"], [{"import": "my_aug", "params": {"unit": "test"}}], True),
        (["sample_count"], "sample_count", True),
        (["sample_count"], 1, False),
        (["seed"], "one", False),
        (["seed"], 32, True),
        (["seed"], -1, False),
        (["extra"], False, False),
        (["extra"], [], False),
    ],
)
def test_validate_schema_dataset(base_cfg, lst, val, result):
    base_cfg["dataset"] = set_nested(base_cfg["dataset"], lst, val)
    _test_validate_schema_pass_fail(base_cfg, result)


@pytest.mark.parametrize(
    "lst,val,result",
    [
        (["network"], "my_net", False),
        (["network"], None, False),
        (["network", "params"], None, True),
        (["network", "import"], None, False),
        (["network"], {"import": "my_net", "params": {"unit": "test"}}, True),
        (["outputs"], "classification", False),
        (["outputs"], [], False),
        (["outputs"], None, False),
        (["outputs"], [{"name": 42, "loss": {"import": "mse"}}], False),
        (["outputs"], [{"loss": {"import": "mse"}}], False),
        (["outputs"], [{"name": "test", "loss": "mse"}], False),
        (["outputs"], [{"name": "test", "loss": {"import": "mse"}}], True),
        (["outputs", 0, "loss_weight"], -1, False),
        (["outputs", 0, "loss_weight"], "one", False),
        (["outputs", 0, "loss_weight"], 7, True),
        (["outputs", 0, "sample_weight_mode"], -1, False),
        (["outputs", 0, "sample_weight_mode"], "temporal", True),
        (["outputs", 0, "sample_weight_mode"], 7, False),
        (["outputs", 0, "metrics"], ["mse", "mse"], False),
        (["outputs", 0, "metrics"], ["mse", {"import": "mse"}], False),
        (["outputs", 0, "name"], None, False),
        (["outputs", 0, "name"], 42, False),
        (["outputs", 0, "name"], "regression", False),
        (["outputs", 1, "name"], "classification", False),
        (["outputs", 0, "loss_weight"], None, False),
        (["outputs", 1, "loss_weight"], None, False),
    ],
)
def test_validate_schema_model(base_cfg, lst, val, result):
    base_cfg["model"] = set_nested(base_cfg["model"], lst, val)
    _test_validate_schema_pass_fail(base_cfg, result)


@pytest.mark.parametrize(
    "lst,val,result",
    [
        (["batch_size"], [32, 64, 128], False),
        (["batch_size"], "32", False),
        (["batch_size"], 7.13, False),
        (["batch_size"], -1, False),
        (["batch_size"], 32, True),
        (["batch_size"], 64, True),
        (["batch_size"], 13, True),
        (["batch_size"], 67, True),
        (["epochs"], [32, 64, 128], False),
        (["epochs"], "32", False),
        (["epochs"], 7.13, False),
        (["epochs"], -1, False),
        (["epochs"], 32, True),
        (["epochs"], 64, True),
        (["epochs"], 13, True),
        (["epochs"], 67, True),
        (["steps"], [32, 64, 128], False),
        (["steps"], "32", False),
        (["steps"], 7.13, False),
        (["steps"], -1, False),
        (["steps"], 32, True),
        (["steps"], 64, True),
        (["steps"], 13, True),
        (["steps"], 67, True),
        (["optimizer"], {"import": "Adam"}, False),
        (["optimizer"], {"import": "Adam", "lr": 1e-5}, False),
        (["optimizer"], {"import": "Adam", "learning_rate": 0}, False),
        (["optimizer", "learning_rate"], {"import": "decay"}, True),
        (["optimizer", "learning_rate"], {"import": "decay", "params": {"a": 1}}, True),
        (["optimizer", "learning_rate"], {"import": "decay", "a": 1}, False),
        (["optimizer", "learning_rate"], {"params": {"a": 1}}, False),
        (["optimizer", "beta"], 0.9, False),
        (["optimizer"], {"import": "RMSProp", "learning_rate": 1e-1}, True),
        (["learning_rate_reducer", "monitor"], None, False),
        (["learning_rate_reducer", "monitor"], 42, False),
        (["learning_rate_reducer", "monitor"], "loss", True),
        (["learning_rate_reducer", "mode"], None, False),
        (["learning_rate_reducer", "mode"], 42, False),
        (["learning_rate_reducer", "mode"], "auto", False),
        (["learning_rate_reducer", "mode"], "min", True),
        (["learning_rate_reducer", "mode"], "max", True),
        (["learning_rate_reducer", "patience"], None, False),
        (["learning_rate_reducer", "patience"], "zero", False),
        (["learning_rate_reducer", "patience"], -1, False),
        (["learning_rate_reducer", "patience"], 1.5, False),
        (["learning_rate_reducer", "patience"], 7, True),
        (["learning_rate_reducer", "factor"], None, False),
        (["learning_rate_reducer", "factor"], "zero", False),
        (["learning_rate_reducer", "factor"], -1, False),
        (["learning_rate_reducer", "factor"], 0.07, True),
        (["learning_rate_reducer", "cooldown"], 5, True),
        (["extra_param"], 42, False),
        (["extra_param"], "7", False),
    ],
)
def test_validate_schema_solver(base_cfg, lst, val, result):
    base_cfg["solver"] = set_nested(base_cfg["solver"], lst, val)
    _test_validate_schema_pass_fail(base_cfg, result)


@pytest.mark.parametrize(
    "lst,val,result",
    [
        (["best_checkpoint"], [], False),
        (["best_checkpoint", "mode"], None, False),
        (["best_checkpoint", "mode"], 42, False),
        (["best_checkpoint", "monitor"], None, False),
        (["best_checkpoint", "monitor"], 42, False),
        (["best_checkpoint", "extra"], "param", False),
        (["best_checkpoint"], {"monitor": "val_acc", "mode": "max"}, True),
        (["best_checkpoint"], {"monitor": "val_regret", "mode": "min"}, True),
        (["tensorboard"], [], False),
        (["tensorboard"], {"batch_size": 42}, True),
        (["train_early_stopping", "monitor"], None, False),
        (["train_early_stopping", "monitor"], 42, False),
        (["train_early_stopping", "monitor"], "loss", True),
        (["train_early_stopping", "mode"], None, False),
        (["train_early_stopping", "mode"], 42, False),
        (["train_early_stopping", "mode"], "auto", False),
        (["train_early_stopping", "mode"], "min", True),
        (["train_early_stopping", "mode"], "max", True),
        (["train_early_stopping", "patience"], None, False),
        (["train_early_stopping", "patience"], "zero", False),
        (["train_early_stopping", "patience"], -1, False),
        (["train_early_stopping", "patience"], 1.5, False),
        (["train_early_stopping", "patience"], 7, True),
        (["train_early_stopping", "min_delta"], None, False),
        (["train_early_stopping", "min_delta"], "zero", False),
        (["train_early_stopping", "min_delta"], -1, False),
        (["train_early_stopping", "min-delta"], 0.07, True),
        (["validation_early_stopping", "monitor"], None, False),
        (["validation_early_stopping", "monitor"], 42, False),
        (["validation_early_stopping", "monitor"], "loss", True),
        (["validation_early_stopping", "mode"], None, False),
        (["validation_early_stopping", "mode"], 42, False),
        (["validation_early_stopping", "mode"], "auto", False),
        (["validation_early_stopping", "mode"], "min", True),
        (["validation_early_stopping", "mode"], "max", True),
        (["validation_early_stopping", "patience"], None, False),
        (["validation_early_stopping", "patience"], "zero", False),
        (["validation_early_stopping", "patience"], -1, False),
        (["validation_early_stopping", "patience"], 1.5, False),
        (["validation_early_stopping", "patience"], 7, True),
        (["validation_early_stopping", "min_delta"], None, False),
        (["validation_early_stopping", "min_delta"], "zero", False),
        (["validation_early_stopping", "min_delta"], -1, False),
        (["validation_early_stopping", "min-delta"], 0.07, True),
    ],
)
def test_validate_schema_services(base_cfg, lst, val, result):
    base_cfg["services"] = set_nested(base_cfg["services"], lst, val)
    _test_validate_schema_pass_fail(base_cfg, result)
