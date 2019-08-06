==============
Barrage Basics
==============

.. contents:: **Table of Contents**:

Welcome to the Barrage package, it may take a while getting used to writing configs
instead of code and or learning the fundamental dataset classes in the short term;
however, the standardization, repeatability, and code reuse should  pay dividends in
the long term.

|Barrage Logo|

.. |Barrage Logo| image:: resources/barrage_logo_small.png

----------
Python API
----------
The Barrage ``python`` API is concise and simple:

.. code-block:: python

  from barrage import BarrageModel

  # Data
  training_dataframe = ...    # pd.DataFrame or list of dicts
  validation_dataframe = ...  # pd.DataFrame or list of dicts
  testing_dataframe = ...     # pd.DataFrame or list of dicts

  # Train a model
  config = {...}
  bm = BarrageModel(artifact_directory)
  bm.train(config, training_dataframe, validation_dataframe)

  # Load a model
  bm = BarrageModel(artifact_directory).load()

  # Score a model
  scores = bm.predict(testing_dataframe)

in both ``BarrageModel.train`` and ``BarrageModel.predict`` the number of ``workers``
and ``max_queue_size`` can be specified for the dataset iterators:

.. code-block:: python

  # To disable multiprocessing: workers = 1
  bm.train(config, training_dataframe, validation_dataframe, workers=10, max_queue_size=20)
  bm.score(test_dataframe, workers=10, max_queue_size=20)


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
