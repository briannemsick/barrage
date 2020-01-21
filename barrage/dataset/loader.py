from typing import Optional

import numpy as np

from barrage import api


class KeySelector(api.RecordLoader):
    """Record loader for directly transforming keys from a record into a data record.

    Args:
        mode: RecordMode, load mode.
        inputs: dict, {input: key or [keys], ...}.
        outputs: dict, {output: key or [keys], ...}.
        sample_weights: dict or None (OPTIONAL), {output: key, ...}.

    Raises:
        TypeError, invalid params.
    """

    def __init__(
        self,
        mode: api.RecordMode,
        inputs: dict,
        outputs: dict,
        sample_weights: Optional[dict] = None,
    ):
        super().__init__(mode)

        self.inputs = inputs
        self.outputs = outputs
        self.sample_weights = sample_weights

        if not isinstance(self.inputs, dict):
            raise TypeError("KeySelector param 'inputs' must be type dict")
        if not isinstance(self.outputs, dict):
            raise TypeError("KeySelector param 'outputs' must be type dict")

        if not (isinstance(self.sample_weights, dict) or self.sample_weights is None):
            raise TypeError("KeySelector 'sample_weights' must be type dict or None")

    def load(self, record: api.Record) -> api.DataRecord:
        """Load a record by selecting keys corresponding to inputs, outputs, and
        maybe sample weights.

        Args:
            record: Record, record.

        Returns:
            DataRecord, data record.
        """

        def _index_dict_to_arr(d, keys):
            if isinstance(keys, list):
                return np.array([d[k] for k in keys])
            else:
                return np.array(d[keys])

        X = {k: _index_dict_to_arr(record, v) for k, v in self.inputs.items()}
        if self.mode == api.RecordMode.TRAIN or self.mode == api.RecordMode.VALIDATION:
            y = {k: _index_dict_to_arr(record, v) for k, v in self.outputs.items()}
            if self.sample_weights is not None:
                w = {k: np.array(record[v]) for k, v in self.sample_weights.items()}
                return (X, y, w)
            else:
                return (X, y)
        else:
            return (X,)
