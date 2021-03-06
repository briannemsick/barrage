==============
Barrage Basics
==============

.. contents:: **Table of Contents**:

Welcome to the Barrage package, it may take a while getting used to writing configs
instead of code and or learning the fundamental dataset classes in the short term;
however, the standardization, repeatability, and code reuse should  pay dividends in
the long term.

----------------
Python Model API
----------------
The Barrage ``python`` Model API is concise and simple:

.. code-block:: python

  from barrage import BarrageModel

  # Data
  training_records = ...    # list of dicts or pandas DataFrame
  validation_records = ...  # list of dicts or pandas DataFrame
  testing_records = ...     # list of dicts or pandas DataFrame

  # Train a model
  config = {...}
  bm = BarrageModel(artifact_directory)
  bm.train(config, training_records, validation_records)

  # Load a model
  bm = BarrageModel(artifact_directory).load()

  # Score a model
  scores = bm.predict(testing_records)

---------------------
Automatic Artifacting
---------------------

Barrage automatically artifacts under the hood based on the user specified artifact directory.

The files and folders written are as follows:

* ``{artifact_dir}/config.json``: human readable copy of the user specified config
  with defaults applied.
* ``{artifact_dir}/config.pkl``: pickled copy of the user specified config with defaults
  applied.
* ``{artifact_dir}/network_params.json``: human readable copy of special params that
  the transformer computed that need to be passed to the network builder (e.g.
  compute vocabulary size from training dataset and pass to the embedding layer).
* ``{artifact_dir}/network_params.json``: pickled copy of the special params from the
  transformer.
* ``{artifact_dir}/training_report.csv``: output metrics vs. epoch report.
* ``{artifact_dir}/best_checkpoint/``: directory for best performing network weights.
* ``{artifact_dir}/resume_checkpoints/``: directory for network weights for each epoch.
* ``{artifact_dir}/TensorBoard/``: directory for ``TensorFlow TensorBoard``.
* ``{artifact_dir}/dataset/``: directory for transformer to save objects for loading
  and saving.
