import os

from click.testing import CliRunner
import pytest

from barrage.console import cli as barrage_cli
from barrage.utils import io_utils
from tests.functional.test_single_output import gen_records


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def data(tmpdir):
    path = os.path.join(tmpdir, "data.json")
    io_utils.save_json(gen_records(42), path)
    return path


@pytest.mark.parametrize(
    "cmd", [["--help"], ["help"], ["train", "--help"], ["predict", "--help"]]
)
def test_help(runner, cmd):
    result = runner.invoke(barrage_cli, cmd)
    assert result.exit_code == 0


def test_train(shared_artifact_dir, runner, data):
    config = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "../functional/config_single_output.json",
    )
    cmd = ["train", config, data, data, "-a", shared_artifact_dir]
    result = runner.invoke(barrage_cli, cmd)
    assert result.exit_code == 0
    assert os.path.isfile(os.path.join(shared_artifact_dir, "config.pkl"))


def test_predict(shared_artifact_dir, runner, data):
    output = os.path.join(shared_artifact_dir, "predictions.json")
    cmd = ["predict", data, shared_artifact_dir, "-o", output]
    result = runner.invoke(barrage_cli, cmd, catch_exceptions=False)
    assert result.exit_code == 0
    assert os.path.isfile(output)
