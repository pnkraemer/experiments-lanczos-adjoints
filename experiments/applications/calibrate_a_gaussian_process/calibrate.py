"""Calibrate an output-independent Gaussian process model on a UCI dataset.

Currently, use either of the following datasets:
* uci_concrete_compressive_strength (small)
* uci_combined_cycle_power_plant (medium)
"""

# todo: make matrix-free
# todo: pivoted cholesky preconditioner
# todo: decide where to go from here

import argparse
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


def data_train_test_split_80_20(X, y, /, *, key):
    """Split a data set into a training and a testing part."""
    ints = jnp.arange(0, len(X))
    ints_shuffled = jax.random.permutation(key, ints)

    idx = len(X) // 5  # 20 percent test data
    train, test = ints_shuffled[:idx], ints_shuffled[idx:]
    return (X[train], y[train]), (X[test], y[test])


def gaussian_process_model(*, shape_in, shape_out) -> tuple[Callable, dict]:
    """Set up the Gaussian process model."""
    parametrise, params_like_kernel = gp.kernel_matern_12(
        shape_in=shape_in, shape_out=shape_out
    )
    params_like_likelihood = jnp.empty(shape_out)
    p_like = {"p_noise_std": params_like_likelihood, "p_kernel": params_like_kernel}
    return parametrise, p_like


def negative_log_likelihood(kernel_func: Callable, X, y):
    """Construct a negative-log-likelihood function."""

    def evaluate(*, p_kernel, p_noise_std):
        """Evaluate the negative log-likelihood of a set of observations."""
        kfun_p = kernel_func(**p_kernel)
        K = kfun_p(X, X.T)

        eye = jnp.eye(len(X))[None, ...]
        K_noisy = K + p_noise_std**2 * eye
        coeffs = jax.vmap(jnp.linalg.solve)(K_noisy, y.T)

        mahalanobis = jax.vmap(jnp.dot)(y.T, coeffs)
        _sign, entropy = jax.vmap(jnp.linalg.slogdet)(K_noisy)
        return jnp.sum(mahalanobis + entropy, axis=0)

    return evaluate


if __name__ == "__main__":
    # Parse the arguments
    # todo: add name_of_run argument to affect the saving directory name
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--name_of_run", type=str, default="")
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument(
        "-n", "--num_points", type=int, default=-1, help="Use -1 for the full dataset"
    )
    parser.add_argument("-k", "--num_epochs", type=int, default=10)
    parser.add_argument(
        "-d", "--dataset", type=str, default="concrete_compressive_strength"
    )
    args = parser.parse_args()

    # Assign them to variables
    print(args, "\n")
    name_of_run = args.name_of_run
    seed = args.seed
    num_points = args.num_points
    num_epochs = args.num_epochs
    dataset_name = args.dataset

    # Initialise the random number generator
    key_ = jax.random.PRNGKey(seed)
    key_data, key_init = jax.random.split(key_, num=2)

    # Load the data
    (X_full, y_full) = exp_util.uci_dataset(dataset_name)
    X_full, y_full = X_full[:num_points], y_full[:num_points]

    # Pre-process the data
    bias = jnp.mean(y_full)
    scale = jnp.std(y_full)
    y_full = (y_full - bias) / scale

    # Split the data into training and testing
    train, test = data_train_test_split_80_20(X_full, y_full, key=key_data)
    (X_train, y_train), (X_test, y_test) = train, test

    # Set up the model
    shape_in = jnp.shape(X_train[0])
    shape_out = jnp.shape(y_train[0])
    kernel, params_like = gaussian_process_model(shape_in=shape_in, shape_out=shape_out)
    params_init = parameters_init(key_init, params_like)

    # Set up the loss function
    loss = negative_log_likelihood(kernel_func=kernel, X=X_train, y=y_train)
    test_nll = negative_log_likelihood(kernel_func=kernel, X=X_test, y=y_test)

    # Optimize (loop until num_epochs is reached or KeyboardInterrupt happens)
    optim = jaxopt.LBFGS(lambda p: loss(**p))
    params_opt, state = params_init, optim.init_state(params_init)
    progressbar = tqdm.tqdm(range(num_epochs))
    progressbar.set_description(f"Loss: {loss(**params_opt):.3F}")
    for _ in progressbar:
        try:
            params_opt, state = optim.update(params_opt, state)
            progressbar.set_description(f"Loss: {loss(**params_opt):.3F}")
        except KeyboardInterrupt:
            break

    # Create a directory for the results
    directory_local = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory_local, exist_ok=True)

    # Save the parameters
    if name_of_run != "":
        name_of_run += "_"
    parameters_save(params_init, directory_local, name=f"{name_of_run}params_init")
    parameters_save(params_opt, directory_local, name=f"{name_of_run}params_opt")

    # Print the results
    print("\nNLL on test set (initial):", test_nll(**params_init))
    print(params_init)
    print("\nNLL on test set (optimized):", test_nll(**params_opt))
    print(params_opt)
