import abc
import enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

Record = Dict[str, Any]
Records = List[Record]
InputRecords = Union[Records, pd.DataFrame]

DataRecord = Tuple[Dict[str, Union[np.ndarray, float]], ...]
BatchDataRecords = Tuple[Dict[str, np.ndarray], ...]

RecordScore = Dict[str, np.ndarray]
BatchRecordScores = List[RecordScore]


class RecordMode(enum.Enum):
    TRAIN = 0
    VALIDATION = 1
    SCORE = 2


class RecordLoader(abc.ABC):
    """Class for loading records into DataRecord.

    Args:
        mode: RecordMode, load mode.
    """

    def __init__(self, mode: RecordMode, **params):
        self.mode = mode

    def __call__(self, record: Record) -> DataRecord:
        return self.load(record)

    @abc.abstractmethod
    def load(self, record: Record) -> DataRecord:  # pragma: no cover
        """Method for loading a record into DataRecord.

        Args:
            record: Record, record.

        Returns:
            DataRecord, data record.
        """
        raise NotImplementedError()


class RecordTransformer(abc.ABC):
    """Class that computes a transform on training data records & applys
    transform to validation and scoring data records (network input), ability
    to pass computed network params to the network builder, and
    ability to apply inverse transforms on record scores (network output).

    Args:
        mode: RecordMode, transform mode.
        loader: RecordLoader, record loader.
    """

    def __init__(self, mode: RecordMode, loader: RecordLoader, **params):
        self.mode = mode
        self.loader = loader
        self._network_params = {}  # type: dict

    @abc.abstractmethod
    def fit(self, records: Records):  # pragma: no cover
        """Fit transform to records.

        Args:
            records: Records, records.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, data_record: DataRecord) -> DataRecord:  # pragma: no cover
        """Apply transform to a data record.

        Args:
            data_record: DataRecord, data record.

        Returns:
            DataRecord, data record.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def postprocess(self, score: RecordScore) -> RecordScore:  # pragma: no cover
        """Postprocess score to undo transform.

        Args:
            score: RecordScore, record output from net.

        Returns:
            RecordScore, postprocessed record output from net.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, path: str):  # pragma: no cover
        """Load transformer.

        Args:
            path: str.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, path: str):  # pragma: no cover
        """Save transformer.

        Args:
            path: str.
        """
        raise NotImplementedError()

    @property
    def network_params(self) -> dict:
        """Special params passed to the network builder."""
        return self._network_params

    @network_params.setter
    def network_params(self, x):
        if not isinstance(x, dict):
            raise TypeError("network_params must be a dict")
        self._network_params = x
