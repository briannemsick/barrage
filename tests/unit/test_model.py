import pytest
import tensorflow as tf
from tensorflow.python.keras import layers, models

from barrage import model


def simple_net(dense_dim=5, input_dim=4, output_dim=3, **params):
    net = models.Sequential()

    net.add(layers.Input(shape=(input_dim,), name="input"))
    net.add(layers.Dense(dense_dim, activation="relu"))
    net.add(layers.Dense(output_dim, activation="linear", name="output"))

    return net


def fake_net():
    return True


def test_build_network():
    cfg_model = {
        "network": {
            "import": "tests.unit.test_model.simple_net",
            "params": {"dense_dim": 7},
        }
    }
    with tf.name_scope("result"):
        net = model.build_network(cfg_model, {"input_dim": 6})
        assert isinstance(net, tf.keras.models.Model)
        result = net.get_config()

    # TODO remove
    tf.keras.backend.reset_uids()

    with tf.name_scope("expected"):
        expected = simple_net(dense_dim=7, input_dim=6).get_config()
    assert result == expected

    cfg_model = {"network": {"import": "tests.unit.test_model.fake_net"}}
    with pytest.raises(TypeError):
        model.build_network(cfg_model, {})

    cfg_model = {"network": {"import": "tests.unit.models.net_bear"}}
    with pytest.raises(ImportError):
        model.build_network(cfg_model, {})


def test_sequential_from_config():
    cfg_model = {
        "network": {
            "import": "barrage.model.sequential_from_config",
            "params": {
                "layers": [
                    {"import": "Input", "params": {"shape": 6, "name": "input"}},
                    {"import": "Dense", "params": {"units": 5, "activation": "relu"}},
                    {
                        "import": "Dense",
                        "params": {
                            "units": 4,
                            "name": "output",
                            "activation": "linear",
                        },
                    },
                ]
            },
        }
    }

    with tf.name_scope("result1"):
        net = model.build_network(cfg_model, {})
        result1 = net.get_config()

    # TODO remove
    tf.keras.backend.reset_uids()

    with tf.name_scope("result2"):
        net = model.sequential_from_config(cfg_model["network"]["params"]["layers"])
        result2 = net.get_config()

    # TODO remove
    tf.keras.backend.reset_uids()

    with tf.name_scope("expected"):
        expected = simple_net(output_dim=4, dense_dim=5, input_dim=6).get_config()

    assert result1 == expected
    assert result2 == expected

    with pytest.raises(KeyError):
        invalid_layers = [{"params": {"shape": 6, "name": "input"}}]
        model.sequential_from_config(invalid_layers)
    with pytest.raises(KeyError):
        invalid_layers = [
            {
                "import": "Input",
                "extra_param": 1,
                "params": {"shape": 6, "name": "input"},
            }
        ]
        model.sequential_from_config(invalid_layers)


def test_build_objective():
    cfg_model = {
        "outputs": [
            {
                "name": "out1",
                "loss": {"import": "categorical_crossentropy"},
                "loss_weight": 0.7,
                "metrics": [
                    {"import": "accuracy"},
                    {
                        "import": "tensorflow.keras.metrics.Precision",
                        "params": {"name": "p"},
                    },
                    {"import": "tf.keras.losses.CategoricalCrossentropy"},
                ],
                "sample_weight_mode": "temporal",
            },
            {
                "name": "out2",
                "loss": {"import": "tensorflow.keras.losses.CategoricalCrossentropy"},
                "loss_weight": 0.3,
            },
        ]
    }
    objective = model.build_objective(cfg_model)
    assert objective["loss"]["out1"] == "categorical_crossentropy"
    assert isinstance(
        objective["loss"]["out2"], tf.keras.losses.CategoricalCrossentropy
    )
    assert objective["metrics"]["out1"][0] == "accuracy"
    assert isinstance(objective["metrics"]["out1"][1], tf.keras.metrics.Precision)
    assert objective["metrics"]["out1"][1].name == "p"
    assert isinstance(
        objective["metrics"]["out1"][2], tf.keras.losses.CategoricalCrossentropy
    )
    assert objective["metrics"]["out2"] == []
    assert objective["loss_weights"] == {"out1": 0.7, "out2": 0.3}
    assert objective["sample_weight_mode"] == {"out1": "temporal", "out2": None}


def single_output_net(compiled):
    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2, name="out_1")(inputs)
    net = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    if compiled:
        net.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    return net


def multi_output_net(compiled):
    inputs = tf.keras.layers.Input(shape=(3,))
    output_1 = tf.keras.layers.Dense(2, name="out_1")(inputs)
    output_2 = tf.keras.layers.Dense(3, activation="softmax", name="out_2")(inputs)
    net = tf.keras.models.Model(inputs=inputs, outputs=[output_1, output_2])

    if compiled:
        net.compile(
            optimizer="Adam",
            loss={"out_1": "mse", "out_2": "categorical_crossentropy"},
            loss_weights={"out_1": 0.6, "out_2": 0.4},
            metrics={"out_1": ["mse", "mae"], "out_2": ["acc"]},
        )
    return net


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize(
    "cfg_model,net_func,result",
    [
        ({"outputs": [{"name": "out_1"}]}, single_output_net, True),
        ({"outputs": [{"name": "out_2"}]}, single_output_net, False),
        ({"outputs": [{"name": "out_1"}, {"name": "out_2"}]}, single_output_net, False),
        ({"outputs": [{"name": "out_1"}, {"name": "out_2"}]}, multi_output_net, True),
        ({"outputs": [{"name": "out_2"}, {"name": "out_1"}]}, multi_output_net, True),
        ({"outputs": [{"name": "out_2"}, {"name": "out_3"}]}, multi_output_net, False),
        (
            {"outputs": [{"name": "out_1"}, {"name": "out_2"}, {"name": "extra"}]},
            multi_output_net,
            False,
        ),
        ({"outputs": [{"name": "out_1"}]}, multi_output_net, False),
        ({"outputs": [{"name": "out_2"}]}, single_output_net, False),
    ],
)
def test_check_output_names(compiled, cfg_model, net_func, result):
    net = net_func(compiled)
    if result:
        model.check_output_names(cfg_model, net)
    else:
        with pytest.raises(ValueError):
            model.check_output_names(cfg_model, net)


# tf.keras API checks
def test_get_metrics_names_single_output():
    net = single_output_net(True)
    result = net.metrics_names
    expected = ["loss", "mae"]
    assert result == expected


def test_get_metrics_names_multi_output():
    net = multi_output_net(True)
    result = net.metrics_names
    expected = [
        "loss",
        "out_1_loss",
        "out_2_loss",
        "out_1_mse",
        "out_1_mae",
        "out_2_acc",
    ]
    assert result == expected


@pytest.mark.parametrize("compiled", [True, False])
def test_get_output_names_single_output(compiled):
    net = single_output_net(compiled)
    result = net.output_names
    expected = ["out_1"]
    assert result == expected


@pytest.mark.parametrize("compiled", [True, False])
def test_get_output_names_multi_output(compiled):
    net = multi_output_net(compiled)
    result = net.output_names
    expected = ["out_1", "out_2"]
    assert result == expected
