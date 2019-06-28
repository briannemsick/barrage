from abc import ABC, abstractmethod

from barrage.dataset import core
from barrage.dataset.loader import RecordLoader


class RecordTransformer(ABC):
    """Class that computes a transform on training data records & applys
    transform to validation and scoring data records (network input), ability
    to pass computed network params to the function that constructs the network, and
    ability to apply inverse transforms on record scores (network output).

    Args:
        mode: RecordMode, transform mode.
        loader: RecordLoader, record loader.
        params: dict.
    """

    def __init__(
        self, mode: core.RecordMode, loader: RecordLoader, params: dict
    ):  # pragma: no cover
        self.mode = mode
        self.loader = loader
        self.params = params
        self._network_params = {}  # type: dict

    @abstractmethod
    def fit(self, records: core.Records):  # pragma: no cover
        """Fit transform to records.

        Args:
            records: Records, records.
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(
        self, data_record: core.DataRecord
    ) -> core.DataRecord:  # pragma: no cover
        """Apply transform to a data record.

        Args:
            data_record: DataRecord, data record.

        Returns:
            DataRecord, data record.
        """
        raise NotImplementedError()

    @abstractmethod
    def postprocess(
        self, score: core.RecordScore
    ) -> core.RecordScore:  # pragma: no cover
        """Postprocess score to undo transform.

        Args:
            score: RecordScore, record output from net.

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

    def fit(self, records: core.Records):
        pass

    def transform(self, data_record: core.DataRecord) -> core.DataRecord:
        return data_record

    def postprocess(self, score: core.RecordScore) -> core.RecordScore:
        return score

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
