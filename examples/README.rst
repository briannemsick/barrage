========
Examples
========

#. ``mnist``: single input, single output, classification. Introduces ``sequential_from_config``.

#. ``UCI Sentiment Labelled Sentences``: single input, single output, text classification. Introduces a text-based ``transformer``.

#. ``iris``: single input, single output, classification. Introduces ``loader``, ``transformer``, and ``augmentor`` (overkill for ``iris`` - purely for example purposes).

#. For a multi input, mult output, samples weights, classification + regression please look
   at ``tests/functional/test_multi_output.py`` and ``tests/functional/config_multi_output.json``.
