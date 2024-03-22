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
pip install .[dev]
```

To install the experiment-related dependencies, run
```commandline
pip install .[experiments, suite_sparse, uci]
```

To install everything and in editable mode, run
```commandline
pip install -e .[dev,experiments,suite_sparse,uci]
```
