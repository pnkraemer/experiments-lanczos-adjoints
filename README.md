# matfree-extensions


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

To install the experiment-related dependencies, run
```commandline
pip install ".[experiments, suitesparse, uci]"
```

To install everything and in editable mode, run
```commandline
pip install -e ".[dev,experiments,suitesparse,uci]"
```


To run all comparisons to GPyTorch (including those in tests/test_gp/),
make sure that PyTorch is installed.

Then, run
```commandline
pip install ".[devtorch]"
```
which is separate from `dev` because it depends on pytorch, not on JAX.


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


# todos:

- run tests without having to rely on (gpy)torch
