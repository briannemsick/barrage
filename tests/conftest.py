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


@pytest.fixture(scope="module")
def shared_artifact_dir(tmpdir_factory):
    return os.path.join(tmpdir_factory.mktemp("shared"), "artifacts")


@pytest.fixture(autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    np.random.seed(7)
