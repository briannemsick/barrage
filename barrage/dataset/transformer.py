from abc import ABC, abstractmethod

import pandas as pd

from barrage.dataset import DataRecordType, RecordLoader, RecordMode, RecordScoreType


class RecordTransformer(ABC):
    """Class that computes a transform on training data records & applys
    transform to validation and scoring data records (network input), ability
    to pass computed network params to the function that constructs the network, and
    ability to apply inverse transforms on record scores (network output).

    Examples uses:
        NLP text classification - fit (vocabulary, max_sequence_length) ->
            (embedding layer, batch_shape).
        Time Series regression - fit mean variance normalization on the
            inputs and outputs, undo normalization on scores.

    Abstract Methods:
        fit: fit a transform to training records to compute params.
        transform:  transform a data record.
        postprocess:  inverse transform on a score.
        load: load transform.
        save: save transform.

    Properties:
        network_params - computed params passed to the network construction

    Args:
        mode: RecordMode, transform mode.
        loader: RecordLoader, record loader.
        params: dict.
    """

    def __init__(
        self, mode: RecordMode, loader: RecordLoader, params: dict
    ):  # pragma: no cover
        self.mode = mode
        self.loader = loader
        self.params = params
        self._network_params = {}  # type: dict

    @abstractmethod
    def fit(self, records: pd.DataFrame):  # pragma: no cover
        """Fit transform to records.

        Args:
            records: pd.DataFrame, data records.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(
        self, data_record: DataRecordType
    ) -> DataRecordType:  # pragma: no cover
        """Apply transform to a data record.

        Args:
            data_record: DataRecordType, data record.

        Returns:
            DataRecordType, data record.
        """
        raise NotImplementedError()

    @abstractmethod
    def postprocess(
        self, score: RecordScoreType
    ) -> RecordScoreType:  # pragma: no cover
        """Postprocess score to undo transform.

        Args:
            score: RecordScoreType, record output from net.

        Returns:
            score.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: str):  # pragma: no cover
        """Load transformer.

        Args:
            path: str.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):  # pragma: no cover
        """Save transformer.

        Args:
            path: str.
        """
        raise NotImplementedError()

    @property
    def network_params(self):
        return self._network_params

    @network_params.setter
    def network_params(self, x):
        if not isinstance(x, dict):
            raise TypeError("network_params must be a dict")
        self._network_params = x


class IdentityTransformer(RecordTransformer):
    """Default transformer that does nothing (identity transform) that ensures
    every dataset has a transformer.
    """

    def fit(self, records: pd.DataFrame):
        pass

    def transform(self, data_record: DataRecordType) -> DataRecordType:
        return data_record

    def postprocess(self, score: RecordScoreType) -> RecordScoreType:
        return score

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass