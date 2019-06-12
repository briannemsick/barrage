===============
Barrage Dataset
===============

.. contents:: **Table of Contents**:


Barrage datasets are built around ``pandas.DataFrame``  and are designed to be flexible, performant,
and powerful data iterators. The underlying data in a Barrage dataset can either be stored as filepaths
or directly inside columns a convenient property for both small and large datasets.

Datasets are comprised of three components:

#. Loaders: load an individual record.

#. Transformers: fit transforms and apply them at batch time.

#. Augmentor: apply a chain of augmentation functions.

----------
High-level
----------

``TensorFlow.Keras`` has a lot of data type API combinations that can be used with ``model.fit``
whether it be ``dicts``, ``numpy arrays``, ``tensors``, ``generators``, ``sequences``, etc..

Barrage uses the ``dicts`` + ``sequences`` API for several key reasons:

#. ``dicts`` are the preferred API - using lists with multi input and multi output is error prone
   and inevitably leads to order mismatch.

#. Everything is a ``sequence`` - irrelevant for small datasets but
   an absolute necessity for large datasets.

#. **Single API** - ``sequence`` regardless of data size and ``dicts`` regardless of single/multi input/output.


**Note**: All ``input`` and ``output`` ``network layers`` **must be named**. The ``output``
layer names are required in the config to associate ``loss`` functions and ``metrics``.
The ``input`` layer names are required for the ``loader``.


~~~~~~~~~~~~~~
System Diagram
~~~~~~~~~~~~~~

Barrage supports 3 modes of data:

.. code-block:: python

  from barrage.dataset import RecordMode


  RecordMode.TRAIN
  RecordMode.VALIDATION
  RecordMode.SCORE

Before deep learning training begins, the transformer does a first pass over the train dataset to fit
transforms:

.. image:: source/transform.png

After deep learning training begins, the transformer applies the transform in conjunction with
loading and augmentation:

.. image:: source/record.png

~~~~~~~~~~
Data Types
~~~~~~~~~~

Data types are critical at each hand-off between components. Here is a summary of the data types:

``Data Record``: a tuple of dictionaries of arrays comprising a single element of a
``TensorFlow.Keras`` batch.

.. code-block:: python

  import numpy as np


  # Example: two inputs, single output
  x = {"network_input_1": np.array([1, 2, 3]), "network_input_2": np.array([4])}
  y = {"network_output_1": np.array([0, 1])}
  w = {"network_output_1": np.array([4])}

  data_record_train = (x, y)  # or (x, y, w) if sample weights
  data_record_validation = (x, y)  # or (x, y, w) if sample weights
  data_record_score = (x,)

``Batch Data Records``: are a stacked tuple of dictionaries of arrays comprising an entire
batch.

.. code-block:: python

  import numpy as np


  # Example: two records, two inputs, single output
  x = {"network_input_1": np.array([[1, 2, 3], [5, 6, 7]]), "network_input_2": np.array([[4], [5]])}
  y = {"network_output_1": np.array([[0, 1], [1, 0]])}
  w = {"network_output_1": np.array([[4],[2]])}

  batch_data_record_train = (x, y)  # or (x, y, w) if sample weights
  batch_data_record_validation = (x, y)  # or (x, y, w) if sample weights
  batch_data_record_score = (x,)

``Record Score``: is a dictionary of arrays comprising a single elements output from
predict.

.. code-block:: python

  import numpy as np


  # Example: single output
  record_score =  {"network_output_1": np.array([0.1, 0.9])}

``Batch Record Scores``: are a list of dictionary of arrays comprising an entire batch output
predict.

.. code-block:: python

  import numpy as np


  # Example: two records, single output
  batch_records_score =  [{"network_output_1": np.array([0.1, 0.9])}, {"network_output_1": np.array([0.7, 0.3])}]

+------------------------------+-----------------+------------------+
| Component Action             | Input Data Type | Output Data Type |
+==============================+=================+==================+
| loader.__call__(...)         | pd.Series       | Data Record      |
+------------------------------+-----------------+------------------+
| transformer.fit(...)         | pd.DataFrame    | None             |
+------------------------------+-----------------+------------------+
| transformer.score(...)       | Data Record     | Data Record      |
+------------------------------+-----------------+------------------+
| transformer.postprocess(...) | Record Score    | Record Score     |
+------------------------------+-----------------+------------------+
| augmentor.__call__(...)      | Data Record     | Data Record      |
+------------------------------+-----------------+------------------+

**Note**: the underlying dataset implementation stacks ``Data Records`` into
``Batch Data Records`` and ``Record Score`` into ``Batch Record Score`` automatically.

**Example Text Classification**:

#. Loader:

   #. Input Data - load a text file or select text column from ``pd.DataFrame``.
   #. Output Data - select label column.

#. Transformer:

   #. Fit - vocabulary to all training records, iterating over records.
   #. Pass - vocabulary size to network builder.
   #. Score - tokenize, pad text sequences, encoding.

**Example Time Series Regression**:

#. Loader:

   #. Input Data - load input stream(s) file(s) or select column(s) from ``pd.DataFrame``.
   #. Output Data - load output stream(s) file(s) or select column(s) from ``pd.DataFrame``.

#. Transformer:

   #. Fit - mean variance normalization to all training records, iterating over records.
   #. Score - mean variance normalize to record.
   #. Postprocess - undo mean variance normalization to record score.

#. Augmentor:

   #. Add Gaussian noise
   #. Phase shift
   #. etc...


------
Loader
------

The loader takes a record (``pd.Series``) and transforms it into a ``Data Record``. This could be
as simple as directly indexing the ``pd.Series`` or loading a filepath stored in the ``pd.Series``.

~~~~~~~~~~
Base Class
~~~~~~~~~~

.. code-block:: python

  from barrage.dataset import RecordLoader

``RecordLoader`` is an abstract base class with properties ``self.mode`` (``RecordMode``) and ``self.params`` from the config.
To write a new ``RecordLoader`` implement the abstract ``load`` method:

.. code-block:: python

  @abstractmethod
  def load(self, record: pd.Series) -> DataRecordType:  # pragma: no cover
      """Method for loading a record into DataRecordType.
      Args:
          record: pd.Series, record.

      Returns:
          DataRecordType, data record.
      """
      raise NotImplementedError()


**Note**: ``RecordLoader`` has a special ``load_all`` method that can be used in fitting the
``transformer``. ``loader.load_all(dataframe)`` returns ``Batch Data Records``.

~~~~~~~~~~~~~~
ColumnSelector
~~~~~~~~~~~~~~

Barrage has a single built-in ``loader``: ``ColumnSelector``.

Params:

.. code:: javascript

  {
    "inputs": {input_layer_name: [columns], ...}
    "outputs": {output_layer_names: [columns], ...}
    "sample_weights": {output_layer_name: column, ...} or None
  }

-----------
Transformer
-----------

The ``transformer`` is responsible for fitting transforms to training data, applying transforms
at batch time, passing network params, and undoing scoring.

~~~~~~~~~~
Base Class
~~~~~~~~~~

.. code-block:: python

  from barrage.dataset import RecordTransformer


``RecordTransformer`` is an abstract base class with properties ``self.mode`` (``RecordMode``) and ``self.params`` from the config.
In addition, it has ``self.loader`` a handle to the ``RecordLoader`` which allows the ``RecordTransformer`` to be agnostic to how
the data was stored by the user. To write a new ``RecordTransformer`` implement the ``fit``, ``transform``,
``postprocess``, ``save``, and ``load`` methods:

.. code-block:: python

  @abstractmethod
  def fit(self, records: pd.DataFrame):  # pragma: no cover
      """Fit transform to records.

      Args:
          records: pd.DataFrame, data records.
      """
      raise NotImplementedError()

  @abstractmethod
  def transform(
      self, data_record: DataRecordType
  ) -> DataRecordType:  # pragma: no cover
      """Apply transform to a data record.

      Args:
          data_record: DataRecordType, data record.

      Returns:
          DataRecordType, data record.
      """
      raise NotImplementedError()

  @abstractmethod
  def postprocess(
      self, score: RecordScoreType
  ) -> RecordScoreType:  # pragma: no cover
      """Postprocess score to undo transform.

      Args:
          score: RecordScoreType, record output from net.

      Returns:
          score.
      """
      raise NotImplementedError()

  @abstractmethod
  def load(self, path: str):  # pragma: no cover
      """Load transformer.

      Args:
          path: str.
      """
      raise NotImplementedError()

  @abstractmethod
  def save(self, path: str):  # pragma: no cover
      """Save transformer.

      Args:
          path: str.
      """
      raise NotImplementedError()

Setting ``self.network_params = dict(...)`` passes the ``network_params`` to the
``network builder`` in addition to the params from the config.

For example:

.. code:: javascript

  "model": {
    "network": {
      "import": "placeholder.net",
      "params": {
        "num_dense": 7,
        "dense_dim": 200
      }
    },
    ...
  }

.. code-block:: python

  # all params = {"num_dense": 7, "dense_dim": 200} + network_params
  network = model.build_network(cfg["model"], transformer.network_params)

~~~~~~~~~~~~~~~~~~~~
IdentityTransformer
~~~~~~~~~~~~~~~~~~~~

The ``IdentityTransformer`` guarantees that every model has a ``transformer`` and is the config default.
The ``fit`` method does nothing, the ``transform`` method returns
the ``Data Record`` unchanged, and the ``postprocess`` method return the ``Data Record``.

---------
Augmentor
---------

The augmentor applies a chain of augmentation functions. The first argument is always the output
of the previous function.

For example consider the config:

.. code:: javascript

    "augmentor": [
      {
        "import": "placeholder.augment_1",
        "params": {
          "hello": "world"
        }
      },
      {
        "import": "placeholder.augment_2",
      },
      {
        "import": "placeholder.augment_3",
        "params": {
          "num": 42
          "s": "foo bar"
        }
      }
    ]

is equivalent to the following code:

.. code-block:: python

  from placeholder import augment_1, augment_2, augment_3


  augmented_record = augment_3(augment_2(augment_1(data_record, hello="world")), num=42, s="foo bar")
