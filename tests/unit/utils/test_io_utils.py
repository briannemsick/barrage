import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from barrage.utils import io_utils


@pytest.fixture()
def sample_dict():
    return {"unit": "test"}


def test_save_json(artifact_path, sample_dict):
    filename = "unit_test.json"
    io_utils.save_json(sample_dict, filename, artifact_path)
    assert os.path.isfile(os.path.join(artifact_path, filename))

    with open(os.path.join(artifact_path, filename), "r") as fn:
        obj = json.load(fn)
    assert obj == sample_dict


def test_save_json_default(artifact_path):
    filename = "unit_test.json"
    sample_dict = {"unit": np.float32(6.0), "test": np.array([1, 2])}
    io_utils.save_json(sample_dict, filename, artifact_path)
    assert os.path.isfile(os.path.join(artifact_path, filename))

    with open(os.path.join(artifact_path, filename), "r") as fn:
        obj = json.load(fn)
    assert obj == {"unit": 6.0, "test": [1, 2]}


def test_load_json(artifact_path, sample_dict):
    filename = "unit_test.json"
    with open(os.path.join(artifact_path, filename), "w") as fn:
        json.dump(sample_dict, fn)
    assert os.path.isfile(os.path.join(artifact_path, filename))

    obj = io_utils.load_json(filename, artifact_path)
    assert obj == sample_dict


def test_save_pickle(artifact_path, sample_dict):
    filename = "unit_test.pkl"
    io_utils.save_pickle(sample_dict, filename, artifact_path)
    assert os.path.isfile(os.path.join(artifact_path, filename))

    with open(os.path.join(artifact_path, filename), "rb") as fn:
        obj = pickle.load(fn)
    assert obj == sample_dict


def test_load_pickle(artifact_path, sample_dict):
    filename = "unit_test.pkl"
    with open(os.path.join(artifact_path, filename), "wb") as fn:
        pickle.dump(sample_dict, fn)
    assert os.path.isfile(os.path.join(artifact_path, filename))

    obj = io_utils.load_pickle(filename, artifact_path)
    assert obj == sample_dict


def test_load_data(artifact_path, sample_dict):
    expected = pd.DataFrame([sample_dict])

    expected.to_json(os.path.join(artifact_path, "unit_test.json"))
    result = io_utils.load_data("unit_test.json", artifact_path)
    assert result.equals(expected)

    expected.to_csv(os.path.join(artifact_path, "unit_test.csv"), index=False)
    result = io_utils.load_data("unit_test.csv", artifact_path)
    assert result.equals(expected)

    with pytest.raises(FileNotFoundError):
        io_utils.load_data("test_unit.42", artifact_path)

    expected.to_json(os.path.join(artifact_path, "unit_test.foo"))
    with pytest.raises(ValueError):
        io_utils.load_data("unit_test.foo", artifact_path)
