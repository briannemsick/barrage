import copy
import os
from typing import List, Union

import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence

from barrage import logger
from barrage.dataset import core
from barrage.dataset.augmentor import RecordAugmentor
from barrage.dataset.loader import RecordLoader
from barrage.dataset.transformer import RecordTransformer
from barrage.utils import import_utils

SEARCH_MODULES = ["barrage.dataset"]


class RecordDataset(Sequence):
    """A tensorflow.keras.utils.Sequence designed to wrap a DataFrame, apply load
    operations, fit & apply transforms, apply data augmentation, and support sampling
    of records.

    At train time: fit transform -> batch -> load -> transform -> augment.

    At validation time: batch -> load -> transform.

    At score time: batch -> load -> transform.

    Args:
        artifact_dir: str, path to artifact directory.
        cfg_dataset: dict, dataset subsection of config.
        records: Union[pd.DataFrame, Records], data records.
        mode: RecordMode, transform mode.
        batch_size: int, batch size.
    """

    def __init__(
        self,
        artifact_dir: str,
        cfg_dataset: dict,
        records: Union[pd.DataFrame, core.Records],
        mode: core.RecordMode,
        batch_size: int,
    ):

        if not isinstance(mode, core.RecordMode):
            raise TypeError("mode must be type RecordMode")

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
        if self.mode == core.RecordMode.TRAIN and sample_count is not None:
            self._sample_inds = convert_sample_count_to_inds(
                [record[sample_count] for record in self.records]
            )
        else:
            self._sample_inds = list(range(self.num_records))
        self.shuffle()

        logger.info(f"Creating record loader")
        loader_cls = import_utils.import_obj_with_search_modules(
            cfg_dataset["loader"]["import"], search_modules=SEARCH_MODULES
        )
        self.loader = loader_cls(
            mode=mode, params=cfg_dataset["loader"].get("params", {})
        )
        if not isinstance(self.loader, RecordLoader):
            raise TypeError(f"loader {self.loader} is not of type RecordLoader")

        logger.info(f"Creating record transformer")
        transformer_cls = import_utils.import_obj_with_search_modules(
            cfg_dataset["transformer"]["import"], search_modules=SEARCH_MODULES
        )
        self.transformer = transformer_cls(
            mode=self.mode,
            loader=self.loader,
            params=cfg_dataset["transformer"].get("params", {}),
        )
        if not isinstance(self.transformer, RecordTransformer):
            raise TypeError(
                f"transformer {self.transformer} is not of type RecordTransformer"
            )

        dataset_dir = os.path.join(artifact_dir, "dataset")
        if self.mode == core.RecordMode.TRAIN:
            logger.info("Creating record augmentor")
            self.augmentor = RecordAugmentor(cfg_dataset["augmentor"])
            logger.info(f"Fitting transform: {self.transformer.__class__.__name__}")
            self.transformer.fit(copy.deepcopy(self.records))
            logger.info(
                f"Transformer network params: {self.transformer.network_params}"
            )
            logger.info("Saving transformer")
            self.transformer.save(dataset_dir)
        else:
            logger.info(f"Loading transform: {self.transformer.__class__.__name__}")
            self.transformer.load(dataset_dir)

    def __len__(self):
        """Number of batches in a sequence."""
        return int(np.ceil(len(self.sample_inds) / float(self.batch_size)))

    def __getitem__(self, ind) -> core.BatchDataRecords:
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

        if self.mode == core.RecordMode.TRAIN:
            lst_data_records = [
                self.augmentor(self.transformer.transform(self.loader(record)))
                for record in batch_records
            ]
        else:
            lst_data_records = [
                self.transformer.transform(self.loader(record))
                for record in batch_records
            ]

        return core.batchify_data_records(lst_data_records)

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
        if (
            self.mode == core.RecordMode.TRAIN
            or self.mode == core.RecordMode.VALIDATION
        ):
            self.sample_order = self.sample_inds
            np.random.shuffle(self.sample_order)
        else:
            self.sample_order = self.sample_inds


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
