# experiments-lanczos-adjoints

This repository contains the experiment code for the paper

> **Gradients of functions of large matrices.**
> _Nicholas Krämer, Pablo Moreno-Muñoz, Hrittik Roy, Søren Hauberg._
> 2024
> Preprint on arXiv: <identifier-coming-soon>.

The paper is currently under review, and the content of this repository subject to change.


## Installation


Before installing this project,
and [after creating & activating your virtual environment](https://realpython.com/python-virtual-environments-a-primer/),
you must install JAX yourself because CPU and GPU backends require different installation commands.
See [here](https://jax.readthedocs.io/en/latest/installation.html) for instructions.
For the small examples, `pip install jax[cpu]` will suffice.
For the bigger experiments, a GPU is helpful.

Then, install the project via
```commandline
pip install .
```

To install the development-related dependencies, run
```commandline
pip install ".[dev]"
```

To install the experiment-related dependencies, first make sure
that JAX and pytorch are installed (JAX has been mentioned above;
for PyTorch, see [here](https://pytorch.org/), then run
```commandline
pip install ".[experiments]"
```


To install everything and in editable mode (recommended!),
first install JAX and PyTorch, and then run
```commandline
pip install -e ".[dev,experiments]"
```

## Running dev-related code

Run tests via
```
pytest
```
or use the pre-defined script
```commandline
make test
```

## Pre-commit hook

after activating your virtual environment, run
```commandline
pre-commit install
```
you only do this once! not every time you activate the environment.


The pre-commit hook is useful, because it implies that we do not accidentally (auto)format each others' code, which would lead to nasty merge conflicts.

## Running code and saving results/figures

Ideally, place experiment scripts into a dedicated subdirectory of experiments/...

E.g., GP related code into experiments/applications/gaussian_process.

Save figures and results in (maybe matching) directories in results/ or figures/.

This could be easier with `matfree_extensions.exp_util.create_matching_directory()`; see the existing scripts for how to use it.
(requires running all scripts from the root, i.e. from where pyproject.toml is.)


## Using code

Look at tests for how to use the library functions.
