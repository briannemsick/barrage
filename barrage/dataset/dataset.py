import os
from typing import List

import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence

from barrage import logger
from barrage.dataset import (
    batchify_data_records,
    BatchDataRecordsType,
    RecordAugmentor,
    RecordLoader,
    RecordMode,
    RecordTransformer,
)
from barrage.utils import import_utils

SEARCH_MODULES = ["barrage.dataset"]


class RecordDataset(Sequence):
    """A tensorflow.keras.utils.Sequence designed to wrap a DataFrame, apply load
    operations, fit & apply transforms, apply data augmentation, and support sampling
    of records.

    Modes:
        TRAIN:  init->fit transform, batch load->transform->augment.
        VALIDATION:  batch->load->transform.
        SCORE: batch->load->transform.

    Args:
        artifact_dir: str, path to artifact directory.
        cfg_dataset: dict, dataset subsection of config.
        records: pd.DataFrame, data records.
        mode: RecordMode, transform mode.
        batch_size: int, batch size.
    """

    def __init__(
        self,
        artifact_dir: str,
        cfg_dataset: dict,
        records: pd.DataFrame,
        mode: RecordMode,
        batch_size: int,
    ):
        """Initialize record dataset: configure loader, transformer and augmentor.

        Args:
            artifact_dir: str, path to artifact directory.
            cfg_dataset: dict, dataset subsection of config.
            records: pd.DataFrame, data records.
            mode: RecordMode, dataset mode.
            batch_size: int, batch size.
        """
        if not isinstance(records, pd.DataFrame):
            raise TypeError("records must be type pd.DataFrame")
        records.reset_index(drop=True, inplace=True)
        if not isinstance(mode, RecordMode):
            raise TypeError("mode must be type RecordMode")

        self.num_records = len(records)
        logger.info(f"Building {mode} dataset with {self.num_records} records")
        self.records = records
        self.mode = mode
        self.batch_size = batch_size

        self.seed = cfg_dataset.get("seed")
        np.random.seed(self.seed)

        sample_count = cfg_dataset.get("sample_count")
        if self.mode == RecordMode.TRAIN and sample_count is not None:
            self._sample_inds = convert_sample_count_to_inds(records[sample_count])
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
        if self.mode == RecordMode.TRAIN:
            logger.info("Creating record augmentor")
            self.augmentor = RecordAugmentor(cfg_dataset["augmentor"])
            logger.info(f"Fitting transform: {self.transformer.__class__.__name__}")
            self.transformer.fit(self.records.copy(deep=True))
            logger.info(
                f"Transformer network params: {self.transformer.network_params}"
            )
            logger.info("Saving transformer")
            self.transformer.save(dataset_dir)
        elif self.mode == RecordMode.VALIDATION or self.mode == RecordMode.SCORE:
            logger.info(f"Loading transform: {self.transformer.__class__.__name__}")
            self.transformer.load(dataset_dir)

    def __len__(self):
        """Number of batches in a sequence."""
        return int(np.ceil(len(self.sample_inds) / float(self.batch_size)))

    def __getitem__(self, ind) -> BatchDataRecordsType:
        """Get a batch by index.

        Args:
            ind: int, batch index.

        Returns:
            BatchDataRecordsType, batch data records.
        """
        batch_inds = self.sample_order[
            ind * self.batch_size : (ind + 1) * self.batch_size
        ]
        batch_records = self.records.iloc[batch_inds].copy(deep=True)

        if self.mode == RecordMode.TRAIN:
            lst_data_records = [
                self.augmentor(self.transformer.transform(self.loader(record)))
                for _, record in batch_records.iterrows()
            ]
        else:
            lst_data_records = [
                self.transformer.transform(self.loader(record))
                for _, record in batch_records.iterrows()
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
        if self.mode == RecordMode.TRAIN or self.mode == RecordMode.VALIDATION:
            self.sample_order = self.sample_inds
            np.random.shuffle(self.sample_order)
        else:
            self.sample_order = self.sample_inds


def convert_sample_count_to_inds(sample_count: pd.Series) -> List[int]:
    """Convert a series of sample counts to a list of sample inds.

    Args:
        sample_count: pd.Series, integer series of sample counts.

    Returns:
        list[int], sample inds.

    Raises:
        TypeError, non-integer pd.Series.
    """
    sample_count = sample_count.apply(lambda x: max(1, x)).astype(int)
    sample_lsts = [[ind] * val for ind, val in sample_count.iteritems()]
    sample_inds = [item for li in sample_lsts for item in li]
    return sample_inds
