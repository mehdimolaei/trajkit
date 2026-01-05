# Contributing to trajkit

Thanks for your interest in improving trajkit! Here’s a concise guide to help you get changes merged smoothly.

## Getting started
- Use Python 3.10+.
- Create a virtualenv: `python -m venv .venv && source .venv/bin/activate`.
- Install dev deps: `pip install -e ".[dev]"`.

## Development workflow
1. Fork/branch from `main`.
2. Make focused changes with clear commits.
3. Run formatting/linting: `ruff check .` and `black .` (line length 100).
4. Run tests: `pytest` (or `pytest tests/path/test_*.py` for targeted runs).
5. Update docs/notebooks if behavior or APIs change.
6. Open a PR describing what/why, mentioning related issues if any.

## Coding style
- Follow existing patterns; prefer small, composable functions.
- Type hints where practical; keep public APIs typed.
- Avoid silent failures—raise clear errors for invalid input.
- Keep logs/prints out of library code; prefer explicit returns/exceptions.

## Docs
- Sphinx sources live in `docs/`; top-level entry is `docs/index.rst`.
- Add or update narrative docs when adding features.
- For tutorials/notebooks, keep them runnable and lightweight; note any data dependencies.

## Tests
- Add or update tests under `tests/` for new functionality or bugfixes.
- Prefer fast, deterministic tests; skip or mark slow/integration as needed.

## Reporting issues
- Include Python/Sphinx versions, OS, and minimal reproduction steps or code.
- For docs issues, note the page path and what’s incorrect or missing.

## Code of conduct
- Be respectful and collaborative. If in doubt, assume good intent and ask clarifying questions.
