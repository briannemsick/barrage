from barrage.dataset import RecordAugmentor


def augment_add(data_record, ind, input_key, value=0):
    data_record[ind][input_key] += value
    return data_record


def augment_mult(data_record, ind, input_key, value=1):
    data_record[ind][input_key] *= value
    return data_record


def test_record_augmentor():
    funcs = [
        {
            "import": "tests.unit.dataset.test_augmentor.augment_add",
            "params": {"input_key": "x1", "ind": 0},
        },
        {
            "import": "tests.unit.dataset.test_augmentor.augment_add",
            "params": {"input_key": "x2", "ind": 0, "value": 2},
        },
        {
            "import": "tests.unit.dataset.test_augmentor.augment_mult",
            "params": {"input_key": "x1", "ind": 0, "value": 5},
        },
        {
            "import": "tests.unit.dataset.test_augmentor.augment_add",
            "params": {"input_key": "y", "ind": 1, "value": 4},
        },
    ]
    augmentor = RecordAugmentor(funcs)

    data_record_1 = ({"x1": 1, "x2": 1}, {"y": 0})
    expected = ({"x1": 5, "x2": 3}, {"y": 4})
    result = augmentor.augment(data_record_1)
    assert result == expected

    data_record_2 = ({"x1": 3, "x2": 2}, {"y": 1})
    expected = ({"x1": 15, "x2": 4}, {"y": 5})
    result = augmentor(data_record_2)
    assert result == expected


def test_record_augmentor_no_funcs():
    funcs = []
    augmentor = RecordAugmentor(funcs)

    data_record_1 = ({"x1": 1, "x2": 1}, {"y": 0})
    expected = ({"x1": 1, "x2": 1}, {"y": 0})
    result = augmentor.augment(data_record_1)
    assert result == expected
    result = augmentor(data_record_1)
    assert result == expected


def test_reduce_compose():
    def add1(x):
        return x + 1

    def mult2(x):
        return x * 2

    func = RecordAugmentor.reduce_compose(mult2, add1)
    assert func(1) == 4
    assert func(2) == 6
    assert func(13) == 28


def test_reduce_compose_no_funcs():
    func = RecordAugmentor.reduce_compose()
    assert func(1) == 1
    assert func(2) == 2
    assert func(13) == 13
