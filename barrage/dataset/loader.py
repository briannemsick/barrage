from abc import ABC, abstractmethod

import numpy as np

from barrage.dataset import core


class RecordLoader(ABC):
    """Class for loading records into DataRecord.

    Args:
        mode: RecordMode, load mode.
        params: dict.
    """

    def __init__(self, mode: core.RecordMode, params: dict):  # pragma: no cover
        self.mode = mode
        self.params = params

    def __call__(self, record: core.Record) -> core.DataRecord:
        return self.load(record)

    @abstractmethod
    def load(self, record: core.Record) -> core.DataRecord:  # pragma: no cover
        """Method for loading a record into DataRecord.

        Args:
            record: Record, record.

        Returns:
            DataRecord, data record.
        """
        raise NotImplementedError()


class KeySelector(RecordLoader):
    """Record loader for transforming keys from a DataFrame into DataRecord.

    Args:
        mode: RecordMode, load mode.
        params: dict,
            inputs: dict, {input: [keys], ...}
            outputs: dict, {output: [keys], ...}
            sample_weights: dict or None (OPTIONAL), {output: key, ...}

    Raises:
        KeyError/TypeError, illegal params.
    """

    def __init__(self, mode: core.RecordMode, params: dict):
        super().__init__(mode, params)

        valid_keys = {"inputs", "outputs", "sample_weights"}
        if not set(params.keys()) <= valid_keys:
            raise KeyError(
                f"Column selector accepts the following params: {valid_keys}"
            )
        if "inputs" not in params:
            raise KeyError("KeySelector required param 'inputs' missing")
        if "outputs" not in params:
            raise KeyError("KeySelector required param 'outputs' missing")

        self.inputs = params["inputs"]
        self.outputs = params["outputs"]
        self.sample_weights = params.get("sample_weights", None)

        if not isinstance(self.inputs, dict):
            raise TypeError("KeySelector param 'inputs' must be type dict")
        if not isinstance(self.outputs, dict):
            raise TypeError("KeySelector param 'outputs' must be type dict")

        if not (isinstance(self.sample_weights, dict) or self.sample_weights is None):
            raise TypeError("KeySelector 'sample_weights' must be type dict or None")

    def load(self, record: core.Record) -> core.DataRecord:
        """Load a record by selecting keys corresponding to inputs, outputs, and
        maybe sample weights.

        Args:
            record: Record, record.

        Returns:
            DataRecord, data record.
        """

        def _index_dict_to_arr(d, lst):
            return np.array([d[v] for v in lst])

        X = {k: _index_dict_to_arr(record, v) for k, v in self.inputs.items()}
        if (
            self.mode == core.RecordMode.TRAIN
            or self.mode == core.RecordMode.VALIDATION
        ):
            y = {k: _index_dict_to_arr(record, v) for k, v in self.outputs.items()}
            if self.sample_weights is not None:
                w = {k: np.array(record[v]) for k, v in self.sample_weights.items()}
                return (X, y, w)
            else:
                return (X, y)
        else:
            return (X,)
