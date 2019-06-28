from typing import List

from barrage.dataset import core
from barrage.utils import import_utils


class RecordAugmentor(object):
    """Class for applying a list of data augmentation functions to a data record.

    Args:
        funcs: list[dict], list of augmentation functions
            {"import": "python_path", "params": {...}}.
    """

    def __init__(self, funcs: List[dict]):
        self.augment_func = reduce_compose(
            *[import_utils.import_partial_wrap_func(f) for f in reversed(funcs)]
        )

    def __call__(self, data_record: core.DataRecord) -> core.DataRecord:
        return self.augment(data_record)

    def augment(self, data_record: core.DataRecord) -> core.DataRecord:
        """Apply augmentation to a train data record.

        Args:
            data_record: DataRecord, data record.

        Returns:
            DataRecord, augmented data record.
        """
        return self.augment_func(data_record)


def reduce_compose(*funcs):
    """Compose a list of functions into a single function."""
    if len(funcs) == 0:
        return lambda x: x

    from functools import reduce

    def _compose2(func1, func2):
        return lambda *a, **kw: func1(func2(*a, **kw))

    return reduce(_compose2, funcs)
