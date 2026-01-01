Installation
============

Quick install from PyPI::

    pip install trajkit

Development setup (with tests/docs)::

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    pip install -r docs/requirements.txt

Optional extras:

- ``pip install "trajkit[viz]"`` for notebook/plotting helpers.
- ``pip install "trajkit[docs]"`` to build the docs locally.
