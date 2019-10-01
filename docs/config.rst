==============
Barrage Config
==============

.. contents:: **Table of Contents**:

Configs serve a central role in Barrage. They are an entire recipe for a deep learning
model - comprised of the components (e.g. etl classes, ``TensorFlow`` metrics,
``TensorFlow`` optimizers, etc...), with auto-configuring defaults and best practices,
and serve as instructions for the underlying Barrage engine to orchestrate training and
scoring.

Configs are broken into four distinct sections:

#. ``dataset``: define how to load, transform, augment a dataset.

#. ``model``: define the model structure, metrics, losses, loss weights, etc...

#. ``solver``: define the optimizer and learning rate scheduling.

#. ``services``: define the underlying artifacting and configure best practices.

**Note**: It is highly recommended (but not required) to use a container
(e.g. ``Docker``) to version the entire environment because it enables the user to
exactly reproduce the environment used to train the model at scoring time
(and or deployment time).

-------
Imports
-------

Components are imported via their ``python paths`` in the config.

An ``import block`` in the config is defined as such:

.. code:: javascript

  {
    "import": str (required python path),
    "params": dict (optional params)
  }

Let's consider several examples where a config import is translated into python code:

* A custom user class:

.. code:: javascript

  {
    "import":  "placeholder.path.MyClass"
    "params": {
      "a": "hello world",
      "b": 42
    }
  }

.. code-block:: python

   from placeholder.path import MyClass


   MyClass(a="hello world", b=42)

* A custom user function:

.. code:: javascript

  {
    "import":  "sample.my_func"
  }

.. code-block:: python

   from sample import my_func


   # params are optional, none passed by the config
   my_func()


The above strategy additionally works for ``TensorFlow`` imports but can lead to
verbose python paths (e.g. ``tensorflow.keras.losses.CategoricalCrossentropy``).
The following import shorthands are adopt for all ``TensorFlow`` imports
(e.g ``metrics``, ``loss``, ``optimizers``, ``schedulers``, etc...):

.. code-block:: python

  # Respect TensorFlow string aliases
  "categorical_crossentropy" == "tensorflow.keras.losses.CategoricalCrossentropy"

  # Search TensorFlow paths automatically
  "Adam" == "tensorflow.python.keras.optimizer_v2.adam.Adam" == "tensorflow.keras.optimizers.Adam"

In addition in the ``dataset`` section of the config, the following import shorthands are
adopted (e.g. ``loaders``, ``transformers``, etc..):

.. code-block:: python

  # Search barrage.dataset paths
  "KeySelector" == "barrage.dataset.KeySelector"

-----------------------
Config Section: dataset
-----------------------

``dataset`` configures the following:

#. loader

#. transformer

#. augmentor

~~~~~~
Schema
~~~~~~

.. code:: javascript

  "dataset": {
    "loader": {
      "import": string,
      "params": dict  // optional
    },
    "transformer": {
      "import": string,
      "params": dict  // optional
    },
    "augmentor": [  // optional
      {
        "import": string,
        "params": dict  // optional
      }
    ],
    "sample_count": string,  //optional
    "seed": int  // optional
  }

~~~~~~~~
Defaults
~~~~~~~~

.. code:: javascript

  "dataset": {
    "transformer": {
      "import": "IdentityTransformer"
    },
    "augmentor": []
  }

~~~~~~~~~
Breakdown
~~~~~~~~~

* ``dataset``: import a class derived from ``barrage.dataset.RecordLoader``.

* ``transformer``: import a class derived from ``barrage.dataset.RecordTransformer``.

* ``augmentor``: list of augmentation functions to import and apply in sequential order.

* ``sample_count``: name of a key that contains integer counts that represent the number of times to
  put a sample in an epoch.

* ``seed``: numpy random seed.

---------------------
Config Section: model
---------------------

``model`` configures the following:

#. network architecture

#. loss functions and loss weights

#. metrics

~~~~~~
Schema
~~~~~~

.. code:: javascript

  "model": {
    "network": {
      "import": string,
      "params": dict  // optional
    },
    "outputs": [
      "name": string,
      "loss": {
        "import": string,
        "params": dict  // optional
      },
      "loss_weight": float, // required if len(outputs) > 1
      "metrics": [  // optional
        {
          "import": string,
          "params": dict  // optional
        }
      ],
      "sample_weight_mode": str //optional
    ]
  }


~~~~~~~~
Defaults
~~~~~~~~

.. code:: javascript

  "model": {}


~~~~~~~~~
Breakdown
~~~~~~~~~

* ``network``: import a function that returns a ``tensorflow.python.keras.Model``.

* ``outputs.name``: string that **must match** an output name from the ``Model`` return by ``network``.

* ``outputs.loss``: import a loss (must be ``v2`` loss class compliant).

* ``outputs.loss_weight``: loss weight for a multi output network.

* ``outputs.metrics``: import a list of metrics (must be ``v2`` metric or loss class compliant).

* ``outputs.sample_weight_mode``: sample weight mode.

----------------------
Config Section: solver
----------------------

``solver`` configures the following:

#. optimizer

#. learning rate scheduling technique

#. batch size

#. epochs

~~~~~~
Schema
~~~~~~

.. code:: javascript

  "solver": {
    "optimizer": {  // optional, all or none
      "import": string,  // required
      "learning_rate": float or import block  // required
      "params": dict  // optional
    },
    "batch_size": int,  // optional
    "epochs": int,  // optional
    "steps": int,  // optional
    "learning_rate_reducer": {
        "monitor": string,
        "mode": "min" or "max",
        "patience": int,
        "factor": float
        // optional additional ReduceLROnPlateau callback  params
    }
  }

**Note**: ``mode =  "auto"`` is not supported.


~~~~~~~~
Defaults
~~~~~~~~

.. code:: javascript

  "solver": {
    "optimizer": {
      "import": "Adam",
      "learning_rate": 1e-3,
      "params": {}
    },
    "batch_size": 32,
    "epochs": 10
  }


~~~~~~~~~
Breakdown
~~~~~~~~~

* ``optimizer``: import a ``TensorFlow`` optimizer (must be compatible with ``v2`` optimizer class).

* ``optimizer.learning_rate``: can be a float or an import block to a schedule (must be compatible with ``v2`` schedule class)

.. code:: javascript

  // float
  "learning_rate": 1e-3

  // import block
  "learning_rate": {
    "import": "ExponentialDecay",
    "params": {
      "initial_learning_rate": 1e-3,
      "decay_steps": 100,
      "decay_rate": 0.99,
    }
  }

* ``batch_size``: batch size.

* ``epochs``: number of epochs to train.

* ``steps``: modify the length of an ``epoch`` to ``steps`` batches. Can be used to shorten or lengthen an epoch.

* ``learning_rate_reducer``: defines params for an ``ReduceLROnPlateua`` callback:

.. code-block:: python

  from tensorflow.python.keras import callbacks


  callbacks.ReduceLROnPlateau(**cfg["solver"]["learning_rate_reducer"])

------------------------
Config Section: services
------------------------

``services`` automatically configures the following best practices with default settings:

#. the best graph should be saved and it should be derived by the performance
   on a validation metric and **not** a training metric (e.g. ``val_loss`` vs. ``loss``)

#. after every checkpoint interval the graph should be saved.

#. ``TensorBoard`` should be automatically setup.

#. if training loss is not changing -> early stop.

#. if the validation metric that is monitored is not changing -> early stop.

**Note**: Early stopping has the potential to prematurely terminate a train even when
``loss`` or ``val_loss`` may continue to improve later (e.g. learning rate scheduling).
To avoid this issue, the defaults have been generously set for a large number of checkpoint
intervals and a very lax improvement condition (near floating point precision).


~~~~~~
Schema
~~~~~~

.. code:: javascript

  "services": {
    {
      "best_checkpoint": {  // optional, all or none
        "monitor": string,
        "mode": "min" or "max"
      },
      "tensorboard": dict,  // optional TensorBoard callback params
      "train_early_stopping": {  // optional, all or none
        "monitor": string,
        "mode": "min" or "max",
        "patience": int,
        "min_delta": float
        // optional additional EarlyStopping callback params
      }
      "validation_early_stopping": {  // optional, all or none
        "monitor": string,
        "mode": "min" or "max",
        "patience": int,
        "min_delta": float
        // optional additional EarlyStopping callback params
      }
    }
  }

**Note**: ``mode =  "auto"`` is not supported.

~~~~~~~~
Defaults
~~~~~~~~

.. code:: javascript

  "services": {
      "best_checkpoint": {
        "monitor": "val_loss",
        "mode": "min"
      },
      "tensorboard": {},
      "train_early_stopping": {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 10,
        "min_delta": 1e-5,
        "verbose": 1
      }
      "validation_early_stopping": {
        "monitor": "val_loss",
        "mode": "min",
        "min_delta": float,
        "min_delta": 1e-5,
        "verbose": 1
      }
    }
  }

~~~~~~~~~
Breakdown
~~~~~~~~~

* ``best_checkpoint``: defines a ``ModelCheckpoint`` callback where ``save_best_only=True``:

.. code-block:: python

  from tensorflow.python.keras import callbacks


  callbacks.ModelCheckpoint(filepath=..., **cfg["services"]["best_checkpoint"], save_best_only=True)

* ``tensorboard``: defines params for a ``TensorBoard`` callback (``log_dir`` preconfigured automatically):

.. code-block:: python

  from tensorflow.python.keras import callbacks


  callbacks.TensorBoard(log_dir=..., **cfg["services"]["tensorboard"])

* ``train_early_stopping``: defines params for an ``EarlyStopping`` callback that must monitor a train metric:

.. code-block:: python

  from tensorflow.python.keras import callbacks


  callbacks.EarlyStopping(**cfg["services"]["train_early_stopping"])

* ``validation_early_stopping``: defines params for an ``EarlyStopping`` callback that must monitor a validation metric:

.. code-block:: python

  from tensorflow.python.keras import callbacks


  callbacks.EarlyStopping(**cfg["services"]["validation_early_stopping"])
