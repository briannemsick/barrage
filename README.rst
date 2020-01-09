=======
Barrage
=======
|Version| |Python| |License| |Build| |Documentation| |Coverage| |Black|

.. |Version| image:: https://img.shields.io/pypi/v/barrage.svg
   :target: https://pypi.org/project/barrage

.. |Python| image:: https://img.shields.io/pypi/pyversions/barrage.svg
   :target: https://www.python.org/downloads/

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/briannemsick/barrage/blob/master/LICENSE

.. |Build| image:: https://travis-ci.com/briannemsick/barrage.svg?branch=master
   :target: https://travis-ci.com/briannemsick/barrage

.. |Documentation|  image:: https://readthedocs.org/projects/barrage/badge/?version=stable
   :target: https://barrage.readthedocs.io/

.. |Coverage| image:: https://codecov.io/gh/briannemsick/barrage/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/briannemsick/barrage

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

Barrage is an opinionated supervised deep learning tool built on top of
``TensorFlow 2.x`` designed to standardize and orchestrate the training and scoring of
complicated models. Barrage is built around a ``JSON`` config and the
``TensorFlow 2.x`` library using the ``Tensorflow.Keras`` API.


Official documentation can be found at: https://barrage.readthedocs.io/

|Barrage Logo|

.. |Barrage Logo| image:: docs/resources/barrage_logo_small.png
   :target: https://barrage.readthedocs.io/

------------------
Guiding Principles
------------------

#. **Minimal Code**: build well-tested, configurable, and reliable config recipes.
   Use custom code only when it is absolutely necessary.

#. **Component Reusability**: decompose deep learning dataset processing into
   fundamental components (e.g. dataset loaders, data transformations,
   augmentation functions) to maximize reuse between models.

#. **Process Automation**: best practices and artifacting are automatically configured
   (e.g. saving best checkpoint, creating TensorBoard, etc...) with defaults that can
   be adjusted in the config.

#. **Standardize API**: takes an opinionated view and selects the production hardened
   variant of the many ``TensorFlow.Keras`` API choices (e.g. data type choices in
   model.fit).

#. **Cross Domain**: handles single/multi input/output networks seamlessly across
   domains (e.g. Computer Vision, Natural Language Processing, Time Series, etc...).

-------------------------
Select Feature Highlights
-------------------------

#. **Single/multi input/output**: flexible across many types of networks.

#. **Loading**: dataset in memory, on disk, in cloud storage, etc ...

#. **Transforms**: fit transforms on a first-pass of the training dataset with the
   ability to:

   #. pass transform params to network builder (e.g. compute vocabulary size ->
      embedding layer).

   #. apply transform at batch time (e.g. mean variance normalization to input).

   #. undo transform after scoring (e.g. undo mean variance normalization to output).

#. **Augmentation**: chain augmentation functions.

#. **Sampling**: change the number of times a sample is selected in an epoch.

As well as standard ``TensorFlow.Keras`` features such as metrics, sample weights, etc...

------------
Installation
------------

**pip**:

.. code-block:: bash

    pip install barrage

**GitHub source**:

.. code-block:: bash

    git clone https://github.com/briannemsick/barrage
    cd barrage
    python setup.py install
