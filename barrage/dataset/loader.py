from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from barrage.dataset import (
    batchify_data_records,
    BatchDataRecordsType,
    DataRecordType,
    RecordMode,
)


class RecordLoader(ABC):
    """Class for loading records into DataRecordType.

    Abstract Methods:
        load: load a record.
        load_all: load all records.

    Args:
        mode: RecordMode, load mode.
        params: dict.
    """

    def __init__(self, mode: RecordMode, params: dict):  # pragma: no cover
        self.mode = mode
        self.params = params

    def __call__(self, record: pd.Series) -> DataRecordType:
        return self.load(record)

    @abstractmethod
    def load(self, record: pd.Series) -> DataRecordType:  # pragma: no cover
        """Method for loading a record into DataRecordType.

        Args:
            record: pd.Series, record.

        Returns:
            DataRecordType, data record.
        """
        raise NotImplementedError()

    def load_all(self, records: pd.DataFrame) -> BatchDataRecordsType:
        """Method for loading all records into a BatchDataRecordsType.

        Args:
            records: pd.DataFrame, records.

        Returns:
            BatchDataRecordsType, all data records.
        """
        return batchify_data_records(
            [self.load(record) for _, record in records.iterrows()]
        )


class ColumnSelector(RecordLoader):
    """Record loader for transforming columns from a DataFrame into DataRecordType.

    Args:
        mode: RecordMode, load mode.
        params: dict,
            inputs: dict, {input: [columns], ...}
            outputs: dict, {output: [columns], ...}
            sample_weights: dict or None (OPTIONAL), {output: column}

    Raises:
        KeyError/TypeError, illegal params.
    """

    def __init__(self, mode: RecordMode, params: dict):
        super().__init__(mode, params)

        valid_keys = {"inputs", "outputs", "sample_weights"}
        if not set(params.keys()) <= valid_keys:
            raise KeyError(
                f"Column selector accepts the following params: {valid_keys}"
            )
        if "inputs" not in params:
            raise KeyError("ColumnSelector required param 'inputs' missing")
        if "outputs" not in params:
            raise KeyError("ColumnSelector required param 'outputs' missing")

        self.inputs = params["inputs"]
        self.outputs = params["outputs"]
        self.sample_weights = params.get("sample_weights", None)

        if not isinstance(self.inputs, dict):
            raise TypeError("ColumnSelector param 'inputs' must be type dict")
        if not isinstance(self.outputs, dict):
            raise TypeError("ColumnSelector param 'outputs' must be type dict")

        if not (isinstance(self.sample_weights, dict) or self.sample_weights is None):
            raise TypeError("ColumnSelector 'sample_weights' must be typedict or None")

    def load(self, record: pd.Series) -> DataRecordType:
        """Load a record by selecting columns corresponding to inputs and outputs.

        Args:
            record: pd.Series, record.

        Returns:
            DataRecordType, data record.
        """
        X = {k: np.array(record[v]) for k, v in self.inputs.items()}
        if self.mode == RecordMode.TRAIN or self.mode == RecordMode.VALIDATION:
            y = {k: np.array(record[v]) for k, v in self.outputs.items()}
            if self.sample_weights is not None:
                w = {k: np.array(record[v]) for k, v in self.sample_weights.items()}
                return (X, y, w)
            else:
                return (X, y)
        else:
            return (X,)


class IdentityLoader(RecordLoader):
    """Special loader that delegates pd.Series -> DataRecordType to the transformer.
    """

    def load(self, record: pd.Series) -> DataRecordType:
        return record  # type: ignore

    def load_all(self, records: pd.DataFrame) -> BatchDataRecordsType:
        return records  # type: ignore
