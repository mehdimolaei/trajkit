<p align="center">
  <img src="docs/_static/logo.png" alt="trajkit logo" width="260">
</p>

# trajkit

Python toolkit for reproducible trajectory (time-series) analytics and flow-field
inference for Brownian and non-Brownian diffusion processes.
See [ACKNOWLEDGMENTS](ACKNOWLEDGMENTS.md).

This repo is part of a collection of statistical and ML tools I’ve written over the years for analyzing equilibrium and nonequilibrium systems. It’s a continuation of what I learned during my graduate and post-graduate work. More recently, I wanted to learn some new Python libraries and experiment with newer methods on top of my earlier work, so this became a “learning project” for building a cleaner, more reusable toolkit. I have a few CDV-related projects in mind and will share them over time as they’re ready. If you notice something that should be credited more clearly, want to contribute, or have suggestions to improve the code or docs, I’d genuinely appreciate an issue or PR.

## Features
- Lightweight `Trajectory` and `TrajectorySet` data model (frame- or time-based)
- Fast Parquet I/O helpers for saving/loading datasets
- Statitical analysis for single trajectories and trajectory sets
- Flow fields inference. 
- Ready-to-adapt notebooks and plotting helpers for publication-quality figures

## Install
Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install trajkit
```

For development:

```bash
pip install -e ".[dev]"
pytest -q
```


## Documentation
Rendered docs live at [trajkit-learn.readthedocs.io](https://trajkit-learn.readthedocs.io/).
Key sections:
- [Getting Started](https://trajkit-learn.readthedocs.io/en/latest/getting_started/)
- [Concepts](https://trajkit-learn.readthedocs.io/en/latest/concepts/)
- [Tools](https://trajkit-learn.readthedocs.io/en/latest/tools/)
- [Tutorials](https://trajkit-learn.readthedocs.io/en/latest/tutorials/)
- [API Reference](https://trajkit-learn.readthedocs.io/en/latest/reference/)

Explore more examples in `examples/notebooks`.

## Contributing
Issues and PRs are welcome. Please see [CONTRIBUTING](CONTRIBUTING.md) for guidelines.

## License
[MIT](LICENSE)
