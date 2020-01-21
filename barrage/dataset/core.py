import copy
import os
from typing import List, Union

import cytoolz
import numpy as np
import pandas as pd
import tensorflow as tf

from barrage import api, logger
from barrage.utils import import_utils

SEARCH_MODULES = ["barrage.dataset"]


class RecordDataset(tf.keras.utils.Sequence):
    """A sequence designed to wrap a DataFrame-like object: apply load
    operations, fit & apply transforms, apply data augmentation, and support sampling
    of records.

    At train time: fit transform -> batch -> load -> transform -> augment.

    At validation time: batch -> load -> transform.

    At score time: batch -> load -> transform.

    Args:
        artifact_dir: str, path to artifact directory.
        cfg_dataset: dict, dataset subsection of config.
        records: api.InputRecords, records.
        mode: RecordMode, transform mode.
        batch_size: int, batch size.
    """

    def __init__(
        self,
        artifact_dir: str,
        cfg_dataset: dict,
        records: api.InputRecords,
        mode: api.RecordMode,
        batch_size: int,
    ):

        if not isinstance(mode, api.RecordMode):
            raise TypeError("mode must be type RecordMode")

        # Standardize InputRecords to Records
        if isinstance(records, pd.DataFrame):
            records.reset_index(drop=True, inplace=True)
            self.records = records.to_dict(orient="records")
        elif all(isinstance(record, dict) for record in records):
            self.records = records
        else:
            raise TypeError("record must be a list of dicts or pandas DataFrame")

        self.num_records = len(records)
        logger.info(f"Building {mode} dataset with {self.num_records} records")
        self.mode = mode
        self.batch_size = batch_size

        self.seed = cfg_dataset.get("seed")
        np.random.seed(self.seed)

        sample_count = cfg_dataset.get("sample_count")
        if self.mode == api.RecordMode.TRAIN and sample_count is not None:
            self._sample_inds = self.convert_sample_count_to_inds(
                [record[sample_count] for record in self.records]
            )
        else:
            self._sample_inds = list(range(self.num_records))
        self.shuffle()

        loader_cls = import_utils.import_obj_with_search_modules(
            cfg_dataset["loader"]["import"], search_modules=SEARCH_MODULES
        )
        self.loader = loader_cls(mode=mode, **cfg_dataset["loader"].get("params", {}))
        if not isinstance(self.loader, api.RecordLoader):
            raise TypeError(f"loader {self.loader} is not of type RecordLoader")

        transformer_cls = import_utils.import_obj_with_search_modules(
            cfg_dataset["transformer"]["import"], search_modules=SEARCH_MODULES
        )
        self.transformer = transformer_cls(
            mode=self.mode,
            loader=self.loader,
            **cfg_dataset["transformer"].get("params", {}),
        )
        if not isinstance(self.transformer, api.RecordTransformer):
            raise TypeError(
                f"transformer {self.transformer} is not of type RecordTransformer"
            )

        dataset_dir = os.path.join(artifact_dir, "dataset")
        if self.mode == api.RecordMode.TRAIN:
            self.augmentor = RecordAugmentor(cfg_dataset["augmentor"])
            logger.info(f"Fitting transform: {self.transformer.__class__.__name__}")
            self.transformer.fit(copy.deepcopy(self.records))
            logger.info(
                f"Transformer network params: {self.transformer.network_params}"
            )
            self.transformer.save(dataset_dir)
        else:
            self.transformer.load(dataset_dir)

    def __len__(self):
        """Number of batches in a sequence."""
        return int(np.ceil(len(self.sample_inds) / float(self.batch_size)))

    def __getitem__(self, ind) -> api.BatchDataRecords:
        """Get a batch by index.

        Args:
            ind: int, batch index.

        Returns:
            BatchDataRecords, batch data records.
        """
        batch_inds = self.sample_order[
            ind * self.batch_size : (ind + 1) * self.batch_size
        ]
        batch_records = copy.deepcopy([self.records[bi] for bi in batch_inds])

        if self.mode == api.RecordMode.TRAIN:
            lst_data_records = [
                self.augmentor(self.transformer.transform(self.loader(record)))
                for record in batch_records
            ]
        else:
            lst_data_records = [
                self.transformer.transform(self.loader(record))
                for record in batch_records
            ]

        return batchify_data_records(lst_data_records)

    def on_epoch_end(self):
        """Shuffle sample_order on epoch end."""
        self.shuffle()

    @property
    def sample_inds(self):
        """Index of records used for batches. Contains repeats if
        self.mode = RecordMode.TRAIN and self.sample_count.
        """
        return self._sample_inds.copy()

    @sample_inds.setter
    def sample_inds(self, x):
        raise ValueError("illegal set on 'sample_inds'")

    def shuffle(self):
        """Shuffle sample_inds to compute sample_order."""
        if self.mode == api.RecordMode.TRAIN or self.mode == api.RecordMode.VALIDATION:
            self.sample_order = self.sample_inds
            np.random.shuffle(self.sample_order)
        else:
            self.sample_order = self.sample_inds

    @staticmethod
    def convert_sample_count_to_inds(sample_count: List[int]) -> List[int]:
        """Convert a list of sample counts to a list of sample inds.

        Args:
            sample_count: list[int], integer list of sample counts.

        Returns:
            list[int], sample inds.
        """
        sample_count = [max(1, int(sc)) for sc in sample_count]
        sample_lsts = [[ind] * val for ind, val in enumerate(sample_count)]
        sample_inds = [item for li in sample_lsts for item in li]
        return sample_inds


class RecordAugmentor(object):
    """Class for applying a list of data augmentation functions to a data record.

    Args:
        funcs: list[dict], list of augmentation functions
            {"import": "python_path", "params": {...}}.
    """

    def __init__(self, funcs: List[dict]):
        self.augment_func = self.reduce_compose(
            *[import_utils.import_partial_wrap_func(f) for f in funcs]
        )

    def __call__(self, data_record: api.DataRecord) -> api.DataRecord:
        return self.augment(data_record)

    def augment(self, data_record: api.DataRecord) -> api.DataRecord:
        """Apply augmentation to a train data record.

        Args:
            data_record: DataRecord, data record.

        Returns:
            DataRecord, augmented data record.
        """
        return self.augment_func(data_record)

    @staticmethod
    def reduce_compose(*funcs):
        """Compose a list of functions into a single function."""
        if len(funcs) == 0:
            return lambda x: x

        from functools import reduce

        def _compose2(func1, func2):
            return lambda *args, **kwargs: func2(func1(*args, **kwargs))

        return reduce(_compose2, funcs)


def batchify_data_records(data_records: List[api.DataRecord]) -> api.BatchDataRecords:
    """Stack a list of DataRecord into BatchRecord. This process converts a list
    of tuples comprising of dicts {str: float/array} into tuples of dict {str: array}.
    Float/array is concatenated along the first dimension. See Example.

    Args:
        data_records: list[DataRecord], list of individual data records.

    Returns:
        BatchDataRecords, batch data records.

    Example:
    ::

        data_record_1 = ({"input_1": 1, "input_2": 2}, {"output_1": 3})
        data_record_2 = ({"input_1": 2, "input_2": 4}, {"output_1": 6})
        batch_data_records = (
            {"input_1": arr([1, 2], "input_2": arr([2, 4])},
            {"output_1": arr([3, 6])}
        )
    """
    batch_data_records = tuple(
        cytoolz.merge_with(np.array, ii) for ii in zip(*data_records)
    )
    return batch_data_records  # type: ignore


def batchify_network_output(
    network_output: Union[np.ndarray, List[np.ndarray]], output_names: List[str]
) -> api.BatchRecordScores:
    """Convert network output scores to BatchRecordScores. This process converts a
    single numpy array or list of numpy arrays into a list of dictionaries. See example.

    Args:
        network_output: union[np.ndarray, list[np.ndarray], network output.

    Returns:
        BatchRecordScores, batch scores.

    Example:
    ::

        network_output == np.array([[1], [2]])
        output_names = ["y"]
        batch_scores = [{"y": np.array([1])}, {"y": np.array([2])}]
    """
    # Handle type inconsistency between outputs of single output/multi networks
    if isinstance(network_output, np.ndarray):
        dict_output = {output_names[0]: network_output}
        num_scores = len(network_output)
    else:
        dict_output = {
            output_names[ii]: network_output[ii] for ii in range(len(output_names))
        }
        num_scores = len(network_output[0])

    scores = [
        {k: v[ii, ...] for k, v in dict_output.items()}  # type: ignore
        for ii in range(num_scores)
    ]

    return scores
