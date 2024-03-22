"""Calibrate a Gaussian process model on the air_quality dataset."""

# todo: use the full time-series!
# todo: use all output features!
# todo: make matrix-free
# todo: load and plot the input labels
# todo: pivoted cholesky preconditioner
# todo: decide where to go from here...

import argparse
import json
import os
import pickle
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import tqdm
from matfree_extensions import exp_util, gp
from tueplots import axes, figsizes, fontsizes

plt.rcParams.update(figsizes.icml2022_full(nrows=2, ncols=4))
plt.rcParams.update(fontsizes.icml2022())
plt.rcParams.update(axes.lines())


def parameters_init(key: jax.random.PRNGKey, param_dict: dict, /) -> dict:
    """Initialise parameters randomly."""
    flat, unflatten = jax.flatten_util.ravel_pytree(param_dict)
    flat_like = jax.random.normal(key, shape=flat.shape)
    return unflatten(flat_like)


def parameters_save(dct: dict, directory: str, /, *, name: str) -> None:
    """Save a parameter dictionary to a file."""
    with open(f"{directory}/{name}.pkl", "wb") as f:
        pickle.dump(dct, f)


def print_dict(dct, *, indent):
    """Pretty-print a dictionary."""
    dct = jax.tree_util.tree_map(float, dct)
    print(json.dumps(dct, sort_keys=True, indent=indent))


def data_subsample(X, y, /, *, key: jax.random.PRNGKey, num: int):
    """Subsample a data set."""
    ints = jnp.arange(0, len(X))
    ints = jax.random.choice(key, ints, replace=False, shape=(num,))
    ints = jnp.sort(ints)
    return X[ints], y[ints]


def data_train_test_split_80_20(X, y, /, *, key):
    """Split a data set into a training and a testing part."""
    ints = jnp.arange(0, len(X))
    ints_shuffled = jax.random.permutation(key, ints)

    idx = len(X) // 5  # 20 percent test data
    train, test = ints_shuffled[:idx], ints_shuffled[idx:]
    return (X[train], y[train]), (X[test], y[test])


def gaussian_process_model() -> tuple[Callable, dict]:
    """Set up the Gaussian process model."""
    # Set up the kernel components
    kernel_matern_12, params_like_matern_12 = gp.kernel_matern_12()
    kernel_matern_32, params_like_matern_32 = gp.kernel_matern_32()
    kernel_periodic, params_like_periodic = gp.kernel_periodic()

    # Initialize the parameters
    params_like_kernel = {
        "matern_12": params_like_matern_12,
        "matern_32": params_like_matern_32,
        "periodic": params_like_periodic,
    }
    params_like_likelihood = 1.0
    p_like = {"p_noise_std": params_like_likelihood, "p_kernel": params_like_kernel}

    # Initialize the parametrized kernel function

    def param(*, matern_12, matern_32, periodic):
        def k(a, b, /):
            k_12 = kernel_matern_12(**matern_12)(a, b)
            k_32 = kernel_matern_32(**matern_32)(a, b)
            k_p = kernel_periodic(**periodic)(a, b)
            return k_12 + k_32 * k_p

        return k

    return param, p_like


def negative_log_likelihood(kernel_func: Callable, X, y):
    """Construct a negative-log-likelihood function."""

    def evaluate(*, p_kernel, p_noise_std):
        """Evaluate the negative log-likelihood of a set of observations."""
        kfun_p = kernel_func(**p_kernel)
        K = kfun_p(X, X.T)

        eye = jnp.eye(len(K))
        coeffs = jnp.linalg.solve(K + p_noise_std**2 * eye, y)

        mahalanobis = y @ coeffs
        _sign, entropy = jnp.linalg.slogdet(K + p_noise_std**2 * eye)
        return mahalanobis + entropy

    return evaluate


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument(
        "-n", "--num_points", type=int, default=100, help="Use -1 for the full dataset"
    )
    parser.add_argument("-k", "--num_epochs", type=int, default=10)
    args = parser.parse_args()

    # Assign them to variables
    print(args, "\n")
    seed = args.seed
    num_points = args.num_points
    num_epochs = args.num_epochs

    # Initialise the random number generator
    key_ = jax.random.PRNGKey(seed)
    key_data, key_init = jax.random.split(key_, num=2)

    # Load the data
    (X_full, y_full) = exp_util.uci_air_quality()
    X_full, y_full = X_full[:num_points], y_full[:num_points]

    # Pre-process the data
    y_full = jnp.log(y_full)
    bias = jnp.mean(y_full)
    scale = jnp.std(y_full)
    y_full = (y_full - bias) / scale

    # Split the data into training and testing
    train, test = data_train_test_split_80_20(X_full[..., None], y_full, key=key_data)
    (X_train, y_train), (X_test, y_test) = train, test

    # Set up the model
    kernel, params_like = gaussian_process_model()
    params_init = parameters_init(key_init, params_like)

    # Set up the loss function
    loss = negative_log_likelihood(kernel_func=kernel, X=X_train, y=y_train)
    test_nll = negative_log_likelihood(kernel_func=kernel, X=X_test, y=y_test)

    # Optimize (loop until num_epochs is reached or KeyboardInterrupt happens)
    optim = jaxopt.LBFGS(lambda p: loss(**p))
    params, state = params_init, optim.init_state(params_init)
    progressbar = tqdm.tqdm(range(num_epochs))
    progressbar.set_description(f"Loss: {loss(**params):.3F}")
    for _ in progressbar:
        try:
            params, state = optim.update(params, state)
            progressbar.set_description(f"Loss: {loss(**params):.3F}")
        except KeyboardInterrupt:
            break

    params_opt = params

    # Create a directory for the results
    directory_local = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory_local, exist_ok=True)

    # Save the parameters
    parameters_save(params_init, directory_local, name="params_init")
    parameters_save(params_opt, directory_local, name="params_opt")

    # Print the results
    print("\nNLL on test set (initial):", test_nll(**params_init))
    print_dict(params_init, indent=4)
    print("\nNLL on test set (optimized):", test_nll(**params_opt))
    print_dict(params_opt, indent=4)
