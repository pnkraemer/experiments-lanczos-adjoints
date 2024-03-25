"""Calibrate an output-independent Gaussian process model on a UCI dataset.

Currently, use either of the following datasets:
* uci_concrete_compressive_strength (small)
* uci_combined_cycle_power_plant (medium)
"""

# todo: make matrix-free
# todo: pivoted cholesky preconditioner
# todo: decide where to go from here

import argparse
import dataclasses
import os
import pickle
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import jaxopt
import tqdm
from matfree_extensions import exp_util, gp


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


@dataclasses.dataclass
class Solver:
    solve: Callable[[jax.Array, jax.Array], jax.Array]
    logdet: Callable[[jax.Array], jax.Array]


def solver_select(which: str, /) -> Solver:
    if which == "dense":

        def solve(A, b):
            return jnp.linalg.solve(A, b)

        def logdet(A):
            return jnp.linalg.slogdet(A)[1]

        return Solver(solve, logdet)

    raise ValueError


def model_log_likelihood(X, y, kernel: Callable, solver: Solver) -> Callable:
    """Construct a negative-log-likelihood function."""

    def evaluate(*, p_kernel, p_noise_std):
        """Evaluate the negative log-likelihood of a set of observations."""
        # Assemble the kernel matrix
        kfun_p = kernel(**p_kernel)
        K = kfun_p(X, X.T)

        # Solve the linear system
        eye = jnp.eye(len(X))[None, ...]
        K_noisy = K + p_noise_std**2 * eye
        coeffs = jax.vmap(solver.solve)(K_noisy, y.T)

        # Compute the log-determinant
        mahalanobis = jax.vmap(jnp.dot)(y.T, coeffs)
        entropy = jax.vmap(solver.logdet)(K_noisy)

        # Combine the terms
        return -jnp.sum(mahalanobis + entropy, axis=0)

    return evaluate


def model_gaussian_process(*, shape_in, shape_out) -> tuple[Callable, dict]:
    """Set up the Gaussian process model."""
    kfun, p_like_kernel = gp.kernel_matern_12(shape_in=shape_in, shape_out=shape_out)
    p_like_likelihood = jnp.empty(shape_out)
    p_like = {"p_noise_std": p_like_likelihood, "p_kernel": p_like_kernel}
    return kfun, p_like


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_of_run", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_points", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="concrete_compressive_strength")
    parser.add_argument("--solver", type=str, default="dense")
    args = parser.parse_args()
    print(args, "\n")

    # Initialise the random number generator
    key_ = jax.random.PRNGKey(args.seed)
    key_data, key_init = jax.random.split(key_, num=2)

    # Load the data
    (X_full, y_full) = exp_util.uci_dataset(args.dataset)
    X_full, y_full = X_full[: args.num_points], y_full[: args.num_points]

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
    kernel, params_like = model_gaussian_process(shape_in=shape_in, shape_out=shape_out)
    params_init = parameters_init(key_init, params_like)

    # Set up the loss function
    solver = solver_select(args.solver)
    loss = model_log_likelihood(X_train, y_train, kernel=kernel, solver=solver)
    test_loss = model_log_likelihood(X_test, y_test, kernel=kernel, solver=solver)

    # Optimize (loop until num_epochs is reached or KeyboardInterrupt happens)
    optim = jaxopt.LBFGS(lambda p: -loss(**p))
    params_opt, state = params_init, optim.init_state(params_init)
    progressbar = tqdm.tqdm(range(args.num_epochs))
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
    name_of_run = args.name_of_run
    if name_of_run != "":
        name_of_run += "_"
    parameters_save(params_init, directory_local, name=f"{name_of_run}params_init")
    parameters_save(params_opt, directory_local, name=f"{name_of_run}params_opt")

    # Print the results
    print("\nNLL on test set (initial):", -test_loss(**params_init))
    print(params_init)
    print("\nNLL on test set (optimized):", -test_loss(**params_opt))
    print(params_opt)
