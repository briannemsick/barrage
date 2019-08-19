import pytest
import tensorflow as tf

from barrage.utils import import_utils


def import_test(unit="test"):
    return unit


@pytest.mark.parametrize(
    "python_path,search_modules,both_cases,expected",
    [
        ("tests.unit.utils.test_import_utils.import_test", None, False, True),
        ("tests.unit.utils.test_import_utils.import_test2", None, False, False),
        ("tests.unit.utils.test_import_utils.Import_test", None, True, False),
        ("tests.unit.utils.test_import_utils.Import_test", None, False, False),
        ("tests.unit.utils.test_import_utils.import_test", ["tests"], False, True),
        (
            "tests.unit.utils.test_import_utils.import_test",
            ["tests.unit.utils.test_import_utils"],
            False,
            True,
        ),
        ("import_test", None, False, False),
        ("import_test", [], False, False),
        ("import_test", ["tests.unit.utils.test_import_utils"], False, True),
        ("import_test", ["tests", "tests.unit.utils.test_import_utils"], False, True),
        ("import_test2", ["tests", "tests.unit.utils.test_import_utils"], False, False),
        ("Import_test", ["tests", "tests.unit.utils.test_import_utils"], True, True),
        ("import_test", ["tests", "tests.unit"], False, False),
    ],
)
def test_import_obj_with_search_modules(
    python_path, search_modules, both_cases, expected
):
    if expected:
        obj = import_utils.import_obj_with_search_modules(
            python_path, search_modules=search_modules, search_both_cases=both_cases
        )
        assert obj() == "test"
    else:
        with pytest.raises(ImportError):
            import_utils.import_obj_with_search_modules(
                python_path, search_modules=search_modules, search_both_cases=both_cases
            )


@pytest.mark.parametrize(
    "python_path,expected,raises",
    [
        ("tests.unit.utils.test_import_utils.import_test", "test", False),
        ("import_test", "import_test", False),
        ("tests.unit.utils.test_import_utils.import_test2", None, True),
    ],
)
def test_import_or_alias(python_path, expected, raises):
    if raises:
        with pytest.raises(ImportError):
            import_utils.import_or_alias(python_path)
    else:
        obj = import_utils.import_or_alias(python_path)
        if isinstance(obj, str):
            assert obj == expected
        else:
            assert obj() == expected


@pytest.mark.parametrize(
    "import_block,expected",
    [
        ({"import": "categorical_crossentropy"}, "categorical_crossentropy"),
        (
            {
                "import": "tensorflow.keras.losses.CategoricalCrossentropy",
                "params": {"name": "cc"},
            },
            tf.keras.losses.CategoricalCrossentropy(name="cc"),
        ),
        ({"import": "tests.unit.test_solver.fake_loss"}, TypeError),
        ({"import": "tests.unit.test_solver.FakeLoss"}, TypeError),
        (
            {"import": "tensorflow.keras.metrics.Precision", "params": {"name": "p"}},
            TypeError,
        ),
        ({"import": "tests.unit.test_solver.fake_metric"}, TypeError),
        ({"import": "tests.unit.test_solver.FakeMetric"}, TypeError),
    ],
)
def test_import_loss(import_block, expected):
    if isinstance(expected, str):
        loss = import_utils.import_loss(import_block)
        assert loss == expected
    elif expected == TypeError:
        with pytest.raises(TypeError):
            import_utils.import_loss(import_block)
    else:
        loss = import_utils.import_loss(import_block)
        assert type(loss) == type(expected)
        assert loss.get_config() == expected.get_config()


@pytest.mark.parametrize(
    "import_block,expected",
    [
        ({"import": "categorical_crossentropy"}, "categorical_crossentropy"),
        (
            {
                "import": "tensorflow.keras.losses.CategoricalCrossentropy",
                "params": {"name": "cc"},
            },
            tf.keras.losses.CategoricalCrossentropy(name="cc"),
        ),
        ({"import": "tests.unit.test_solver.fake_loss"}, TypeError),
        ({"import": "tests.unit.test_solver.FakeLoss"}, TypeError),
        ({"import": "accuracy"}, "accuracy"),
        (
            {"import": "tensorflow.keras.metrics.Precision", "params": {"name": "p"}},
            tf.keras.metrics.Precision(name="p"),
        ),
        ({"import": "tests.unit.test_solver.fake_metric"}, TypeError),
        ({"import": "tests.unit.test_solver.FakeMetric"}, TypeError),
    ],
)
def test_import_metric(import_block, expected):
    if isinstance(expected, str):
        metric = import_utils.import_metric(import_block)
        assert metric == expected
    elif expected == TypeError:
        with pytest.raises(TypeError):
            import_utils.import_metric(import_block)
    else:
        metric = import_utils.import_metric(import_block)
        assert type(metric) == type(expected)
        assert metric.get_config() == expected.get_config()
