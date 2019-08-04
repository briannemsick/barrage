===============================
GitHub Issues and Pull Requests
===============================

Contributions are both welcome and encouraged. With that in mind here are some
contribution guidelines.

-----------
Bug Reports
-----------

#. Search issues for similiar issue(s). Avoid raising redudant issue reports.

#. Provide the requested information in the issue template: system information,
   clear description, and code & config to reproduce the issue.

#. Consider raising a pull request to fix the bug.

------------
New Features
------------

#. New features will be considered on a case by case basis. Barrage is and will be
   focused on battle-hardened "production ready" features. Bleeding edge research
   ideas will rarely be incorporated.

#. Please provide a code snippet and or example that demonstrates the use case and
   the API of the proposed feature


-------------
Pull Requests
-------------

~~~~~~~~~~~~~~~~~
Development Setup
~~~~~~~~~~~~~~~~~

Please setup your development environment with the following steps:


.. code-block:: bash

  # Clone the repository
  git https://github.com/briannemsick/barrage
  cd barrage

  # Install the test requiremeents
  pip install -e .[tests]

  # Setup pre-commit hooks
  pre-commit install


Please run ``lint``, ``type hint``, and ``test`` before raising a pull request.

``lint``:

.. code-block:: bash

  black . --check
  flake8
  isort --check

``type hint``:

.. code-block:: bash

  find . -name "*.py" | xargs mypy

``test``:

.. code-block:: bash

  python -m pytest --cov=barrage --cov-config=setup.cfg tests/

To build the ``Read the Docs`` locally:

.. code-block:: bash

  cd docs
  pip install -r requirements.txt
  rm barrage.*
  sphinx-apidoc -f -o . ../barrage
  make html

Upload to PyPi (project owner only):

.. code-block:: bash

  python setup.py sdist bdist_wheel
  python -m twine upload dist/*

~~~~~~~~~~~~~~~~~~~~~~
Raising a Pull Request
~~~~~~~~~~~~~~~~~~~~~~

#. Please follow the `Google Python Style Guide <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>`_.

#. All new code requires docstrings, type hints, and tests.

#. Pull requests should have links to corresponding issues, a label, and a clear concise description.
