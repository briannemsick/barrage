import numpy as np
import pandas as pd
import pytest

from barrage.dataset import ColumnSelector, IdentityLoader, RecordMode


@pytest.fixture
def record():
    return pd.Series({"a": 1, "b": 2, "c": "three", "d": 4, "e": "five"})


@pytest.fixture
def records():
    return pd.DataFrame([{"x1": 1, "x2": 2, "y1": 3}, {"x1": 4, "x2": 5, "y1": 6}])


def test_column_selector(record):
    params = {
        "inputs": {"x1": ["a", "b"], "x2": ["b", "c"]},
        "outputs": {"y1": ["d"], "y2": ["e", "d"]},
    }

    cs = ColumnSelector(RecordMode.TRAIN, params)
    result = cs.load(record)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0]["x1"].tolist() == [1, 2]
    assert result[0]["x2"].tolist() == [2, "three"]
    assert len(result[1]) == 2
    assert result[1]["y1"].tolist() == [4]
    assert result[1]["y2"].tolist() == ["five", 4]

    cs = ColumnSelector(RecordMode.VALIDATION, params)
    result = cs.load(record)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0]["x1"].tolist() == [1, 2]
    assert result[0]["x2"].tolist() == [2, "three"]
    assert len(result[1]) == 2
    assert result[1]["y1"].tolist() == [4]
    assert result[1]["y2"].tolist() == ["five", 4]

    cs = ColumnSelector(RecordMode.SCORE, params)
    result = cs(record)
    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0]["x1"].tolist() == [1, 2]
    assert result[0]["x2"].tolist() == [2, "three"]


def test_column_selector_sample_weights(record):
    params = {
        "inputs": {"x1": ["a", "b"], "x2": ["b", "c"]},
        "outputs": {"y1": ["d"], "y2": ["e", "d"]},
        "sample_weights": {"y1": "a", "y2": "b"},
    }

    cs = ColumnSelector(RecordMode.TRAIN, params)
    result = cs.load(record)
    assert len(result) == 3
    assert len(result[0]) == 2
    assert result[0]["x1"].tolist() == [1, 2]
    assert result[0]["x2"].tolist() == [2, "three"]
    assert len(result[1]) == 2
    assert result[1]["y1"].tolist() == [4]
    assert result[1]["y2"].tolist() == ["five", 4]
    assert len(result[2]) == 2
    assert result[2]["y1"].tolist() == 1
    assert result[2]["y2"].tolist() == 2

    cs = ColumnSelector(RecordMode.VALIDATION, params)
    result = cs.load(record)
    assert len(result) == 3
    assert len(result[0]) == 2
    assert result[0]["x1"].tolist() == [1, 2]
    assert result[0]["x2"].tolist() == [2, "three"]
    assert len(result[1]) == 2
    assert result[1]["y1"].tolist() == [4]
    assert result[1]["y2"].tolist() == ["five", 4]
    assert len(result[2]) == 2
    assert result[2]["y1"].tolist() == 1
    assert result[2]["y2"].tolist() == 2

    cs = ColumnSelector(RecordMode.SCORE, params)
    result = cs(record)
    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0]["x1"].tolist() == [1, 2]
    assert result[0]["x2"].tolist() == [2, "three"]


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
def test_column_selector_invalid_values(mode, inputs, outputs, sample_weights, err):
    params = {"inputs": inputs, "outputs": outputs, "sample_weights": sample_weights}
    with pytest.raises(err):
        ColumnSelector(mode, params)


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
@pytest.mark.parametrize(
    "params",
    [
        {"inputs": {"x": ["x"]}, "outputs": {"y": ["y"]}, "new_column": "z"},
        {"inputs": {"x": ["x"]}},
        {"outputs": {"y": ["y"]}},
    ],
)
def test_column_selector_invalid_keys(mode, params):
    with pytest.raises(KeyError):
        ColumnSelector(mode, params)


def _assert_batch_equal(b1, b2):
    """Helper function to compare BatchDataRecords."""
    assert len(b1) == len(b2)
    for ii in range(len(b1)):
        assert set(b1[ii].keys()) == set(b2[ii].keys())
        for key in b1[ii].keys():
            np.testing.assert_array_equal(b1[ii][key], b2[ii][key])


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_load_all(mode, records):
    params = {"inputs": {"x": ["x1", "x2"]}, "outputs": {"y": ["y1"]}}
    cs = ColumnSelector(mode, params)

    expected = ({"x": np.array([[1, 2], [4, 5]])}, {"y": np.array([[3], [6]])})
    if mode == RecordMode.SCORE:
        expected = (expected[0],)
    _assert_batch_equal(cs.load_all(records), expected)


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_identity_loader(mode, record, records):
    il = IdentityLoader(mode, {})
    assert il(record).equals(record)
    assert il.load_all(records).equals(records)
