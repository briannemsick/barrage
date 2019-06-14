import pytest
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

from barrage import solver

opt_adam = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.8)
cfg_adam = {
    "optimizer": {"import": "Adam", "learning_rate": 0.1, "params": {"beta_1": 0.8}}
}
opt_ftrl = tf.keras.optimizers.Ftrl(learning_rate=0.3)
cfg_ftrl = {"optimizer": {"import": "Ftrl", "learning_rate": 0.3}}


@pytest.mark.parametrize(
    "cfg_solver,python_path,expected",
    [
        (cfg_adam, "Adam", opt_adam),
        (cfg_adam, "adam", opt_adam),
        (cfg_adam, "tensorflow.python.keras.optimizer_v2.adam.Adam", opt_adam),
        (cfg_adam, "tensorflow.python.keras.optimizer_v2.adam.Adam", opt_adam),
        (cfg_adam, "tf.keras.optimizers.Adam", opt_adam),
        (cfg_adam, "tensorflow.keras.optimizers.Adam", opt_adam),
        (cfg_adam, "tests.unit.test_solver.FakeOptimizer", TypeError),
        (cfg_adam, "unit.test", ImportError),
        (cfg_ftrl, "Ftrl", opt_ftrl),
        (cfg_ftrl, "ftrl", opt_ftrl),
        (cfg_ftrl, "tensorflow.python.keras.optimizer_v2.ftrl.Ftrl", opt_ftrl),
        (cfg_ftrl, "tf.keras.optimizers.Ftrl", opt_ftrl),
        (cfg_ftrl, "tensorflow.keras.optimizers.Ftrl", opt_ftrl),
    ],
)
def test_build_optimizer(cfg_solver, python_path, expected):
    cfg_solver["optimizer"]["import"] = python_path
    if isinstance(expected, optimizer_v2.OptimizerV2):
        opt = solver.build_optimizer(cfg_solver)
        assert type(opt) == type(expected)
        assert opt.get_config() == expected.get_config()
    else:
        with pytest.raises(expected):
            solver.build_optimizer(cfg_solver)


def test_build_optimizer_with_schedule():
    schedule_sgd = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1, decay_steps=100, decay_rate=0.99
    )
    opt_sgd = tf.keras.optimizers.SGD(learning_rate=schedule_sgd)
    cfg_sgd = {
        "optimizer": {
            "import": "SGD",
            "learning_rate": {
                "import": "ExponentialDecay",
                "params": {
                    "initial_learning_rate": 0.1,
                    "decay_steps": 100,
                    "decay_rate": 0.99,
                },
            },
        }
    }
    opt = solver.build_optimizer(cfg_sgd)
    assert opt_sgd.get_config() == opt.get_config()


def test_create_learning_rate_reducer():
    cfg_solver = {
        "learning_rate_reducer": {
            "monitor": "val_loss",
            "mode": "min",
            "patience": 5,
            "factor": 0.1,
            "verbose": 0,
        }
    }
    metrics_names = ["loss"]
    result = solver.create_learning_rate_reducer(cfg_solver, metrics_names)
    expected = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=5, factor=0.1, verbose=1
    )
    assert type(result) == type(expected)
    vars_result = vars(result)
    vars_expected = vars(expected)
    vars_result.pop("monitor_op")
    vars_expected.pop("monitor_op")
    assert vars_result == vars_expected

    cfg_solver = {
        "learning_rate_reducer": {
            "monitor": "val_acc",
            "mode": "max",
            "patience": 5,
            "factor": 0.1,
        }
    }
    metrics_names = ["loss"]
    with pytest.raises(ValueError):
        solver.create_learning_rate_reducer(cfg_solver, metrics_names)


class FakeOptimizer(object):
    def __init__(self, **params):
        pass


def fake_loss():
    return 1.0


class FakeLoss(object):
    def __init__(self, **params):
        pass


def fake_metric():
    return 1.0


class FakeMetric(object):
    def __init__(self, **params):
        pass
