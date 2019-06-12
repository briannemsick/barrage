import os

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def artifact_dir(tmpdir):
    return os.path.join(tmpdir, "artifact")


@pytest.fixture
def artifact_path(tmpdir):
    return str(tmpdir)


@pytest.fixture(autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    np.random.seed(7)
