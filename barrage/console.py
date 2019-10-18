import click

from barrage import BarrageModel
from barrage.utils import io_utils


@click.group(name="barrage")
def cli():
    pass


@cli.command()
@click.pass_context
def help(context):
    click.echo(context.parent.get_help())


@cli.command("train")
@click.argument("config", type=click.Path(exists=True))
@click.argument("train-data", type=click.Path(exists=True))
@click.argument("validation-data", type=click.Path(exists=True))
@click.option(
    "-a",
    "--artifact-dir",
    default="artifacts",
    type=click.Path(),
    help="location to save artifacts",
)
def train(config, train_data, validation_data, artifact_dir):
    """Barrage deep learning train.

    Supported filetypes:

        1. .csv

        2. .json

    Args:

        config: filepath to barrage config [REQUIRED].

        train-data: filepath to train data [REQUIRED].

        validation-data: filepath to validation data [REQUIRED].

    Note: artifact-dir cannot already exist.
    """
    cfg = io_utils.load_json(config)
    records_train = io_utils.load_data(train_data)
    records_validation = io_utils.load_data(train_data)
    BarrageModel(artifact_dir).train(cfg, records_train, records_validation)


@cli.command("predict")
@click.argument("score-data", type=click.Path(exists=True))
@click.argument("artifact-dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    default="scores.json",
    type=click.Path(),
    help="output filepath for scores",
)
def predict(score_data, artifact_dir, output):
    """Barrage deep learning predict.

    Supported filetypes:

        1. .csv

        2. .json

    Args:

        score-data: filepath to score data [REQUIRED].

        artifact-dir: location to load artifacts [REQUIRED].
    """
    records_score = io_utils.load_data(score_data)
    scores = BarrageModel(artifact_dir).predict(records_score)
    io_utils.save_json(scores, output)
