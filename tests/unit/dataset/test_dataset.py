import os

import numpy as np
import pandas as pd
import pytest

from barrage.api import RecordMode, RecordTransformer
from barrage.dataset import RecordDataset
from barrage.utils import io_utils


@pytest.fixture
def dataset_dir(tmpdir):
    os.mkdir(os.path.join(tmpdir, "dataset"))
    return str(tmpdir)


@pytest.fixture
def records():
    """Records used for all tests with known properties."""
    num_samples = 100
    x1 = np.arange(num_samples, dtype=np.float32)
    x2 = np.arange(num_samples, dtype=np.float32) % 5
    y1 = np.arange(num_samples, dtype=np.float32)
    y2 = np.arange(num_samples, dtype=np.float32)[::-1] % 5
    sample_count = np.random.randint(1, 5, size=num_samples)
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "sample_count": sample_count}
    )


@pytest.fixture
def base_cfg_dataset():
    return {
        "loader": {
            "import": "KeySelector",
            "params": {
                "inputs": {"input_1": ["x1", "x2"]},
                "outputs": {"output_1": ["y1"]},
            },
        },
        "transformer": {"import": "IdentityTransformer"},
        "augmentor": [],
    }


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_dataset_init_basics(dataset_dir, base_cfg_dataset, records, mode):
    # Exploit the fact identity transformer does not save params
    batch_size = 32
    ds = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=base_cfg_dataset,
        records=records,
        mode=mode,
        batch_size=batch_size,
    )

    assert ds.num_records == len(records)
    assert ds.records == records.to_dict(orient="records")
    assert ds.mode == mode
    assert ds.batch_size == batch_size
    assert hasattr(ds, "loader")
    assert hasattr(ds, "transformer")
    if mode == RecordMode.TRAIN:
        assert hasattr(ds, "augmentor")
    else:
        assert not hasattr(ds, "augmentor")
    assert ds.transformer.network_params == {}


@pytest.mark.parametrize("sample_count", [None, "sample_count"])
@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
@pytest.mark.parametrize("batch_size", [1, 13, 32, 128, 269])
def test_dataset_length(
    dataset_dir, base_cfg_dataset, records, mode, sample_count, batch_size
):
    base_cfg_dataset["sample_count"] = sample_count
    ds = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=base_cfg_dataset,
        records=records,
        mode=mode,
        batch_size=batch_size,
    )

    if mode == RecordMode.TRAIN and sample_count is not None:
        num_samples = records[sample_count].sum()
        assert len(ds) == int(np.ceil(num_samples / float(batch_size)))
    else:
        assert len(ds) == int(np.ceil(len(records) / float(batch_size)))


@pytest.mark.parametrize("sample_count", [None, "sample_count"])
@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_dataset_sample_inds(
    dataset_dir, base_cfg_dataset, records, mode, sample_count
):
    base_cfg_dataset["sample_count"] = sample_count
    batch_size = 32
    ds = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=base_cfg_dataset,
        records=records,
        mode=mode,
        batch_size=batch_size,
    )

    if mode == RecordMode.TRAIN and sample_count is not None:
        assert ds.sample_inds == RecordDataset.convert_sample_count_to_inds(
            records[sample_count]
        )
    else:
        assert ds.sample_inds == list(range(len(records)))


@pytest.mark.parametrize("sample_count", [None, "sample_count"])
@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_dataset_sample_order(
    dataset_dir, base_cfg_dataset, records, mode, sample_count
):
    base_cfg_dataset["sample_count"] = sample_count
    base_cfg_dataset["seed"] = 13
    batch_size = 32
    ds = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=base_cfg_dataset,
        records=records,
        mode=mode,
        batch_size=batch_size,
    )

    if mode == RecordMode.TRAIN or mode == RecordMode.VALIDATION:
        assert ds.sample_order != ds.sample_inds
        assert sorted(ds.sample_order) == ds.sample_inds

        sample_order_1 = ds.sample_order
        ds.shuffle()
        sample_order_2 = ds.sample_order
        assert sample_order_1 != sample_order_2
        assert sorted(sample_order_1) == sorted(sample_order_2)
        ds.on_epoch_end()
        sample_order_3 = ds.sample_order
        assert sample_order_2 != sample_order_3
        assert sorted(sample_order_2) == sorted(sample_order_3)
    else:
        assert ds.sample_order == ds.sample_inds

        sample_order_1 = ds.sample_order
        ds.shuffle()
        sample_order_2 = ds.sample_order
        assert sample_order_1 == sample_order_2
        ds.on_epoch_end()
        sample_order_3 = ds.sample_order
        assert sample_order_2 == sample_order_3


class SimpleTransformer(RecordTransformer):
    def fit(self, records):
        num_records = len(records)
        data_record = self.loader(records[0])
        network_params = {
            "num_inputs": len(data_record[0]),
            "num_outputs": len(data_record[1]),
        }

        sum_input = np.zeros_like(data_record[0][self.params["input_key"]])
        sum_output = np.zeros_like(data_record[1][self.params["output_key"]])

        for ind in range(num_records):
            data_record = self.loader.load(records[ind])
            sum_input += data_record[0][self.params["input_key"]]
            sum_output += data_record[1][self.params["output_key"]]

        mean_input = sum_input / float(num_records)
        mean_output = sum_output / float(num_records)

        self.obj = {"mean_input": mean_input, "mean_output": mean_output}
        self.network_params = network_params

    def transform(self, data_record):
        data_record[0][self.params["input_key"]] -= self.obj["mean_input"]
        if self.mode == RecordMode.TRAIN or self.mode == RecordMode.VALIDATION:
            data_record[1][self.params["output_key"]] -= self.obj["mean_output"]
        return data_record

    def postprocess(self, score):
        score[self.params["output_key"]] += self.obj["mean_output"]
        return score

    def save(self, path):
        io_utils.save_pickle(self.obj, "obj.pkl", path)

    def load(self, path):
        self.obj = io_utils.load_pickle("obj.pkl", path)


@pytest.fixture
def transformer_cfg_dataset():
    return {
        "loader": {
            "import": "KeySelector",
            "params": {
                "inputs": {"input_1": ["x1", "x2"]},
                "outputs": {"output_1": ["y1"], "output_2": ["y2"]},
            },
        },
        "transformer": {
            "import": "tests.unit.dataset.test_dataset.SimpleTransformer",
            "params": {"input_key": "input_1", "output_key": "output_2"},
        },
        "augmentor": [
            {
                "import": "tests.unit.dataset.test_augmentor.augment_add",
                "params": {"input_key": "input_1", "ind": 0, "value": 3},
            },
            {
                "import": "tests.unit.dataset.test_augmentor.augment_mult",
                "params": {"input_key": "output_2", "ind": 1, "value": 5},
            },
        ],
    }


@pytest.mark.parametrize("sample_count", [None, "sample_count"])
@pytest.mark.parametrize("mode", [RecordMode.VALIDATION, RecordMode.SCORE])
def test_dataset_init_transformers(
    dataset_dir, transformer_cfg_dataset, records, mode, sample_count
):
    def _assert_dict_array_equal(d1, d2):
        assert len(d1) == len(d2)
        for k in d1.keys():
            np.testing.assert_array_equal(d1[k], d2[k])

    transformer_cfg_dataset["sample_count"] = sample_count
    batch_size = 32
    ds_train = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=transformer_cfg_dataset,
        records=records,
        mode=RecordMode.TRAIN,
        batch_size=batch_size,
    )

    ref_obj = {"mean_input": np.array([49.5, 2]), "mean_output": np.array([2])}
    ref_network_params = {"num_inputs": 1, "num_outputs": 2}

    _assert_dict_array_equal(ds_train.transformer.obj, ref_obj)
    assert ds_train.transformer.network_params == ref_network_params

    ds_non_train = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=transformer_cfg_dataset,
        records=records,
        mode=mode,
        batch_size=batch_size,
    )
    _assert_dict_array_equal(ds_non_train.transformer.obj, ref_obj)


@pytest.mark.parametrize("mode", [RecordMode.VALIDATION, RecordMode.SCORE])
def test_dataset_non_train_before_train(
    dataset_dir, transformer_cfg_dataset, records, mode
):
    batch_size = 32
    with pytest.raises(FileNotFoundError):
        RecordDataset(
            artifact_dir=dataset_dir,
            cfg_dataset=transformer_cfg_dataset,
            records=records,
            mode=mode,
            batch_size=batch_size,
        )


def _assert_batch_equal(b1, b2):
    """Helper function to compare BatchRecordsType."""
    assert len(b1) == len(b2)
    for ii in range(len(b1)):
        assert set(b1[ii].keys()) == set(b2[ii].keys())
        for key in b1[ii].keys():
            np.testing.assert_array_equal(b1[ii][key], b2[ii][key])


def _to_list(r, b):
    return r.copy(deep=True).to_dict(orient="records") if b else r.copy(deep=True)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("sample_count", [None, "sample_count"])
@pytest.mark.parametrize("batch_size", [1, 13, 32, 128, 269])
def test_dataset_train(
    dataset_dir, transformer_cfg_dataset, records, batch_size, sample_count, as_list
):
    transformer_cfg_dataset["sample_count"] = sample_count
    ds_train = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=transformer_cfg_dataset,
        records=_to_list(records, as_list),
        mode=RecordMode.TRAIN,
        batch_size=batch_size,
    )

    for ind in range(len(ds_train)):
        result = ds_train[ind]

        # Compute expected
        sample_inds = ds_train.sample_order[ind * batch_size : (ind + 1) * batch_size]
        batch_records = records.iloc[sample_inds]
        mX = np.array([49.5, 2])
        X = batch_records[["x1", "x2"]].values.reshape(len(sample_inds), 2)
        X -= mX
        X += 3
        y1 = batch_records["y1"].values.reshape(len(sample_inds), 1)
        y2 = batch_records["y2"].values.reshape(len(sample_inds), 1) % 5 - 2
        y2 *= 5
        expected = ({"input_1": X}, {"output_1": y1, "output_2": y2})

        _assert_batch_equal(result, expected)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("batch_size", [1, 13, 32, 128, 269])
def test_dataset_validation(
    dataset_dir, transformer_cfg_dataset, records, batch_size, as_list
):

    # Fit training dataset
    RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=transformer_cfg_dataset,
        records=_to_list(records, as_list),
        mode=RecordMode.TRAIN,
        batch_size=batch_size,
    )

    ds_val = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=transformer_cfg_dataset,
        records=_to_list(records, as_list),
        mode=RecordMode.VALIDATION,
        batch_size=batch_size,
    )

    for ind in range(len(ds_val)):
        result = ds_val[ind]

        # Compute expected
        sample_inds = ds_val.sample_order[ind * batch_size : (ind + 1) * batch_size]
        batch_records = records.iloc[sample_inds]
        mX = np.array([49.5, 2])
        X = batch_records[["x1", "x2"]].values.reshape(len(sample_inds), 2)
        X -= mX
        y1 = batch_records["y1"].values.reshape(len(sample_inds), 1)
        y2 = batch_records["y2"].values.reshape(len(sample_inds), 1) % 5 - 2
        expected = ({"input_1": X}, {"output_1": y1, "output_2": y2})

        _assert_batch_equal(result, expected)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("batch_size", [1, 13, 32, 128, 269])
def test_dataset_score(
    dataset_dir, transformer_cfg_dataset, records, batch_size, as_list
):
    # Fit training dataset
    RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=transformer_cfg_dataset,
        records=_to_list(records, as_list),
        mode=RecordMode.TRAIN,
        batch_size=batch_size,
    )

    ds_score = RecordDataset(
        artifact_dir=dataset_dir,
        cfg_dataset=transformer_cfg_dataset,
        records=_to_list(records, as_list),
        mode=RecordMode.SCORE,
        batch_size=batch_size,
    )

    for ind in range(len(ds_score)):
        result = ds_score[ind]

        # Compute expected
        sample_inds = ds_score.sample_order[ind * batch_size : (ind + 1) * batch_size]
        batch_records = records.iloc[sample_inds]
        mX = np.array([49.5, 2])
        X = batch_records[["x1", "x2"]].values.reshape(len(sample_inds), 2)
        X -= mX
        expected = ({"input_1": X},)

        _assert_batch_equal(result, expected)


@pytest.mark.parametrize(
    "s,result",
    [([1, 4, 2], [0, 1, 1, 1, 1, 2, 2]), ([2, 1, 1, 3], [0, 0, 1, 2, 3, 3, 3])],
)
def test_convert_sample_count_to_inds(s, result):
    assert RecordDataset.convert_sample_count_to_inds(s) == result
