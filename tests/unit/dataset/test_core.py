import numpy as np

from barrage.dataset import batchify_data_records, batchify_network_output


def _assert_batch_data_records_equal(b1, b2):
    """Helper function to compare BatchRecordsType."""
    assert len(b1) == len(b2)
    for ii in range(len(b1)):
        assert set(b1[ii].keys()) == set(b2[ii].keys())
        for key in b1[ii].keys():
            np.testing.assert_array_equal(b1[ii][key], b2[ii][key])


def test_batchify_data_records_singleton_input():
    # batch size = 1, train/validation
    data_records = [({"i1": np.array([1, 2])}, {"o1": np.array([0])}, {"o1": 1})]
    result = batchify_data_records(data_records)
    expected = (
        {"i1": np.array([[1, 2]])},
        {"o1": np.array([[0]])},
        {"o1": np.array([1])},
    )
    _assert_batch_data_records_equal(result, expected)
    # batch_size = 1, score
    data_records = [({"x1": np.array([1, 2])},)]
    result = batchify_data_records(data_records)
    expected = ({"x1": np.array([[1, 2]])},)
    _assert_batch_data_records_equal(result, expected)


def test_batchify_data_records():
    # Single input, single output, train/validation
    data_records = [
        ({"i1": np.array([1, 2])}, {"o1": np.array([0])}, {"o1": 1}),
        ({"i1": np.array([3, 4])}, {"o1": np.array([1])}, {"o1": 2}),
    ]
    result = batchify_data_records(data_records)
    expected = (
        {"i1": np.array([[1, 2], [3, 4]])},
        {"o1": np.array([[0], [1]])},
        {"o1": np.array([1, 2])},
    )
    _assert_batch_data_records_equal(result, expected)

    # Single input, single output, score
    data_records = [
        ({"x1": np.array([1, 2])},),
        ({"x1": np.array([3, 4])},),
        ({"x1": np.array([5, 6])},),
    ]
    result = batchify_data_records(data_records)
    expected = ({"x1": np.array([[1, 2], [3, 4], [5, 6]])},)
    _assert_batch_data_records_equal(result, expected)

    # Multi input, multi output, train/validation
    data_records = [
        (
            {"i1": np.array([1, 2]), "i2": np.array([[3], [4]])},
            {"o1": np.array([0]), "o2": np.array([1, 2])},
            {"o1": 1, "o2": np.array([2, 3])},
        ),
        (
            {"i1": np.array([5, 6]), "i2": np.array([[7], [8]])},
            {"o1": np.array([3]), "o2": np.array([4, 5])},
            {"o1": 4, "o2": np.array([5, 6])},
        ),
    ]
    result = batchify_data_records(data_records)
    expected = (
        {"i1": np.array([[1, 2], [5, 6]]), "i2": np.array([[[3], [4]], [[7], [8]]])},
        {"o1": np.array([[0], [3]]), "o2": np.array([[1, 2], [4, 5]])},
        {"o1": np.array([1, 4]), "o2": np.array([[2, 3], [5, 6]])},
    )
    _assert_batch_data_records_equal(result, expected)

    # Multi input, multi output, score
    data_records = [
        ({"x1": np.array([0, 1]), "x2": np.array([[1], [0]])},),
        ({"x1": np.array([1, 2]), "x2": np.array([[2], [1]])},),
        ({"x1": np.array([3, 4]), "x2": np.array([[4], [3]])},),
    ]
    result = batchify_data_records(data_records)
    expected = (
        {
            "x1": np.array([[0, 1], [1, 2], [3, 4]]),
            "x2": np.array([[[1], [0]], [[2], [1]], [[4], [3]]]),
        },
    )
    _assert_batch_data_records_equal(result, expected)


def _assert_batch_scores_equal(s1, s2):
    """Helper function to compare BatchScoresType."""
    assert len(s1) == len(s2)
    for ii in range(len(s1)):
        assert set(s1[ii].keys()) == set(s2[ii].keys())
        for key in s1[ii].keys():
            np.testing.assert_array_equal(s1[ii][key], s2[ii][key])


def test_batchify_network_output_singleton_input():
    # Single output
    network_output = np.array([[1]])
    output_names = ["o1"]
    result = batchify_network_output(network_output, output_names)
    expected = [{"o1": np.array([1])}]
    _assert_batch_scores_equal(result, expected)

    # Multi output
    network_output = [np.array([[1]]), np.array([[2]])]
    output_names = ["o1", "o2"]
    result = batchify_network_output(network_output, output_names)
    expected = [{"o1": np.array([1]), "o2": np.array([2])}]
    _assert_batch_scores_equal(result, expected)


def test_batchify_network_output():
    # Single output
    network_output = np.array([[1], [2]])
    output_names = ["o1"]
    result = batchify_network_output(network_output, output_names)
    expected = [{"o1": np.array([1])}, {"o1": np.array([2])}]
    _assert_batch_scores_equal(result, expected)

    # Multi output
    network_output = [np.array([[1], [2]]), np.array([[3], [4]])]
    output_names = ["o1", "o2"]
    result = batchify_network_output(network_output, output_names)
    expected = [
        {"o1": np.array([1]), "o2": np.array([3])},
        {"o1": np.array([2]), "o2": np.array([4])},
    ]
    _assert_batch_scores_equal(result, expected)
