import numpy as np
import pandas as pd
import pytest

from barrage.dataset import IdentityTransformer, RecordMode


@pytest.mark.parametrize(
    "mode", [RecordMode.TRAIN, RecordMode.VALIDATION, RecordMode.SCORE]
)
def test_identity_transformer(artifact_path, mode):
    records = pd.DataFrame([{"x_1": 1, "x_2": 0, "y": 0}, {"x_1": 0, "x_2": 1, "y": 1}])

    loader = None  # IdentityTransformer doesn't need a loader
    transformer = IdentityTransformer(mode, loader, {"test": "param"})
    assert transformer.params == {"test": "param"}

    # Fit
    assert transformer.network_params == {}
    transformer.fit(records)
    assert transformer.network_params == {}

    # Transform params
    transformer.network_params = {"unit": "test"}
    assert transformer.network_params == {"unit": "test"}
    with pytest.raises(TypeError):
        transformer.network_params = [1, 2]

    # Transform
    X = {"x": np.array([1, 0])}
    y = {"y": np.array([0])}
    if mode == RecordMode.SCORE:
        data_record = (X,)
    else:
        data_record = (X, y)

    result = transformer.transform(data_record)
    assert result == data_record

    # Score
    score = np.random.randint(0, 10, size=(10, 10))
    np.testing.assert_array_equal(transformer.postprocess(score), score)
