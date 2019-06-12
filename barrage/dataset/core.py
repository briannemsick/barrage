import enum
from typing import Dict, List, Tuple, Union

from cytoolz import merge_with
import numpy as np


BatchDataRecordsElementType = Dict[str, np.array]
BatchDataRecordsType = Union[
    Tuple[BatchDataRecordsElementType],
    Tuple[BatchDataRecordsElementType, BatchDataRecordsElementType],
    Tuple[
        BatchDataRecordsElementType,
        BatchDataRecordsElementType,
        BatchDataRecordsElementType,
    ],
]
DataRecordElementType = Dict[str, Union[np.array, float, int]]
DataRecordType = Union[
    Tuple[DataRecordElementType],
    Tuple[DataRecordElementType, DataRecordElementType],
    Tuple[DataRecordElementType, DataRecordElementType, DataRecordElementType],
]
RecordScoreType = Dict[str, np.array]
BatchRecordScoresType = List[RecordScoreType]


class RecordMode(enum.Enum):
    TRAIN = 0
    VALIDATION = 1
    SCORE = 2


def batchify_data_records(data_records: List[DataRecordType]) -> BatchDataRecordsType:
    """Stack a list of DataRecordType into BatchRecordType. This process converts a list
    of tuples comprising of dicts {str: float/array} into tuples of dict {str: array}.
    Float/array is concatenated along the first dimension. See Example.

    Args:
        data_records: list[DataRecordType], list of individual data records.

    Returns:
        BatchDataRecordsType, batch data records.

    Example:
        data_record_1 = ({"input_1": 1, "input_2": 2}, {"output_1": 3})
        data_record_1 = ({"input_1": 2, "input_2": 4}, {"output_1": 6})
        batch_data_records = (
            {"input_1": arr([1, 2], "input_2": arr([2, 4])},
            {"output_1": arr([3, 6])}
        )
    """
    batch_data_records = tuple(merge_with(np.array, ii) for ii in zip(*data_records))
    return batch_data_records  # type: ignore


def batchify_network_output(
    network_output: Union[np.array, List[np.array]], output_names: List[str]
) -> BatchRecordScoresType:
    """Convert network output scores to BatchRecordScoresType. This process converts a
    single numpy array or list of numpy arrays into a list of dictionaries. See example.

    Args:
        network_output: union[np.array, list[np.array], network output.

    Returns:
        BatchRecordScoresType, batch scores.

    Example:
        network_output == np.array([[1], [2]]), output_names = ["y"]
        batch_scores = [{"y": np.array([1])}, {"y": np.array([2])}]
    """
    # Handle type inconsistency out single output/multi output networks
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
