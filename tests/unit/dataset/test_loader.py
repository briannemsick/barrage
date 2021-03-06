import numpy as np
import pytest

from barrage.api import RecordMode
from barrage.dataset import KeySelector


@pytest.fixture
def record():
    return {"a": 1, "b": 2, "c": "three", "d": 4, "e": "five"}


@pytest.fixture
def record_arr():
    return {"x": np.zeros((28, 28, 1)), "y": 7}


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_key_selector(mode, record):
    # no sample weights
    params = {
        "inputs": {"x1": ["a", "b"], "x2": ["c"]},
        "outputs": {"y1": ["d"], "y2": ["e"]},
    }
    ks = KeySelector(mode, **params)

    result = ks.load(record)
    expected = (
        {"x1": np.array([1, 2]), "x2": np.array(["three"])},
        {"y1": np.array([4]), "y2": np.array(["five"])},
    )
    if mode == RecordMode.SCORE:
        expected = (expected[0],)
    _assert_batch_equal(result, expected)

    # sample weights
    params = {
        "inputs": {"x1": ["a", "b"], "x2": ["c"]},
        "outputs": {"y1": ["d"], "y2": ["e"]},
        "sample_weights": {"y1": "a", "y2": "b"},
    }
    ks = KeySelector(mode, **params)

    result = ks(record)
    expected = (
        {"x1": np.array([1, 2]), "x2": np.array(["three"])},
        {"y1": np.array([4]), "y2": np.array(["five"])},
        {"y1": 1, "y2": 2},
    )
    if mode == RecordMode.SCORE:
        expected = (expected[0],)
    _assert_batch_equal(result, expected)


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_key_selector_arr(mode, record_arr):
    params = {"inputs": {"img": "x"}, "outputs": {"label": "y"}}
    ks = KeySelector(mode, **params)

    result = ks.load(record_arr)
    expected = ({"img": np.zeros((28, 28, 1))}, {"label": np.array([7])})
    if mode == RecordMode.SCORE:
        expected = (expected[0],)
    _assert_batch_equal(result, expected)


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
@pytest.mark.parametrize(
    "inputs,outputs,sample_weights,err",
    [
        ({"x": ["x"]}, "y", None, TypeError),
        ("x", {"y": ["y"]}, None, TypeError),
        ({"x": ["x"]}, {"y": ["y"]}, 15, TypeError),
    ],
)
def test_key_selector_invalid_values(mode, inputs, outputs, sample_weights, err):
    params = {"inputs": inputs, "outputs": outputs, "sample_weights": sample_weights}
    with pytest.raises(err):
        KeySelector(mode, **params)


def _assert_batch_equal(b1, b2):
    """Helper function to compare BatchDataRecords."""
    assert len(b1) == len(b2)
    for ii in range(len(b1)):
        assert set(b1[ii].keys()) == set(b2[ii].keys())
        for key in b1[ii].keys():
            np.testing.assert_array_equal(b1[ii][key], b2[ii][key])
