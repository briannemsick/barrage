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
    "cfg_solver,python_path,result",
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
def test_build_optimizer(cfg_solver, python_path, result):
    cfg_solver["optimizer"]["import"] = python_path
    if isinstance(result, optimizer_v2.OptimizerV2):
        opt = solver.build_optimizer(cfg_solver)
        assert type(opt) == type(result)
        assert vars(opt) == vars(result)
    else:
        with pytest.raises(result):
            solver.build_optimizer(cfg_solver)


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
    assert isinstance(result, tf.keras.callbacks.ReduceLROnPlateau)
    assert isinstance(expected, tf.keras.callbacks.ReduceLROnPlateau)
    vars_result = vars(result)

    # TODO investigate the monitor op
    del vars_result["monitor_op"]
    vars_expected = vars(expected)
    del vars_expected["monitor_op"]
    assert vars_result == vars(expected)

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


def simple_learning_rate_schedule(epoch, lr, alpha=1, beta=1):
    return lr * alpha * beta


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.2), (1.0, 2.0)])
def test_learning_rate_scheduler(alpha, beta):
    cfg_solver = {
        "learning_rate_scheduler": {
            "import": "tests.unit.test_solver.simple_learning_rate_schedule",
            "params": {"alpha": alpha, "beta": beta},
        }
    }
    result = solver.create_learning_rate_scheduler(cfg_solver)
    assert isinstance(result, tf.keras.callbacks.LearningRateScheduler)
    epoch = 2
    lr = 5
    assert result.schedule(epoch, lr) == lr * alpha * beta


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
