===========
Barrage CLI
===========

.. contents:: **Table of Contents**:

Barrage has a very simple command line interface ``cli``.

::

  barrage --help
  Usage: barrage [OPTIONS] COMMAND [ARGS]...

  Options:
    --help  Show this message and exit.

  Commands:
    predict  Barrage deep learning predict.
    train    Barrage deep learning train.

-----
train
-----
::

  barrage train --help
  Usage: barrage train [OPTIONS] CONFIG TRAIN_DATA VALIDATION_DATA

    Barrage deep learning train.

    Supported filetypes:

        1. .csv

        2. .json

    Args:

        config: filepath to barrage config [REQUIRED].

        train-data: filepath to train data [REQUIRED].

        validation-data: filepath to validation data [REQUIRED].

    Note: artifact-dir cannot already exist.

  Options:
    -a, --artifact-dir PATH  location to save artifacts
    --help                   Show this message and exit.


-------
predict
-------
::

  barrage predict --help
  Usage: barrage predict [OPTIONS] SCORE_DATA ARTIFACT_DIR

    Barrage deep learning predict.

    Supported filetypes:

        1. .csv

        2. .json

    Args:

        score-data: filepath to score data [REQUIRED].

        artifact-dir: location to load artifacts [REQUIRED].

  Options:
    -o, --output PATH  output filepath for scores
    --help             Show this message and exit.
