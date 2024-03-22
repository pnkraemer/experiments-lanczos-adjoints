"""Calibrate a Gaussian process model on the air_quality dataset."""

# todo: use the full time-series!
# todo: use all output features!
# todo: make matrix-free
# todo: train-test split
# todo: reduce the evaluation down to a single quantity (NMLL/RMSE on a test set?)
# todo: load and plot the input labels
# todo: pivoted cholesky preconditioner
# todo: decide where to go from here...

import functools
import json
import os
import pickle

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


def parameters_init(key, param_dict: dict, /):
    flat, unflatten = jax.flatten_util.ravel_pytree(param_dict)
    flat_like = jax.random.normal(key, shape=flat.shape)
    return unflatten(flat_like)


def parameters_save(param_dict: dict, direct, name):
    with open(f"{direct}/{name}.pkl", "wb") as f:
        pickle.dump(param_dict, f)


def print_dict(dct, *, indent):
    dct = jax.tree_util.tree_map(float, dct)
    print(json.dumps(dct, sort_keys=True, indent=indent))


def data_subsample(X, y, /, *, key, num):
    ints = jnp.arange(0, len(X))
    ints = jax.random.choice(key, ints, replace=False, shape=(num,))
    ints = jnp.sort(ints)
    return X[ints], y[ints]


def gaussian_process_model():
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
    params_like_noise = 1.0
    p_like = {"noise": params_like_noise, "kernel": params_like_kernel}

    # Initialize the parametrized kernel function

    def param(*, matern_12, matern_32, periodic):
        def k(a, b, /):
            k_12 = kernel_matern_12(**matern_12)(a, b)
            k_32 = kernel_matern_32(**matern_32)(a, b)
            k_p = kernel_periodic(**periodic)(a, b)
            return k_12 + k_32 * k_p

        return k

    return param, p_like


def negative_log_likelihood(parameters_and_noise, /, *, kernel_fun, X, y):
    parameters, noise_std = (
        parameters_and_noise["kernel"],
        parameters_and_noise["noise"],
    )
    kernel_fun_p = kernel_fun(**parameters)
    K = kernel_fun_p(X, X.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, y)

    mahalanobis = y @ coeffs
    _sign, entropy = jnp.linalg.slogdet(K + noise_std**2 * eye)
    return mahalanobis + entropy


if __name__ == "__main__":
    seed = 3
    num_pts = 100

    # Initialise the random number generator
    key_ = jax.random.PRNGKey(seed)
    key_data, key_init = jax.random.split(key_, num=2)

    # Load and subsample the dataset
    (X, y) = exp_util.uci_air_quality()
    X = X[..., None]  # (N, d) shape
    X, y = data_subsample(X[:num_pts], y[:num_pts], key=key_data, num=num_pts)

    # Pre-process the data
    y = jnp.log(y)
    bias = jnp.mean(y)
    scale = jnp.std(y)
    y = (y - bias) / scale

    # Set up the model
    kernel, params_like = gaussian_process_model()
    params_init = parameters_init(key_init, params_like)

    # Set up the loss function
    loss = functools.partial(negative_log_likelihood, kernel_fun=kernel, X=X, y=y)

    # Optimize
    optim = jaxopt.LBFGS(loss)
    params, state = params_init, optim.init_state(params_init)
    for _ in tqdm.tqdm(range(100)):
        params, state = optim.update(params, state)

    params_opt = params

    # Create a directory for the results
    directory = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory, exist_ok=True)

    # Save the parameters
    parameters_save(params_init, directory, "params_init")
    parameters_save(params_opt, directory, "params_opt")

    # Print the parameters
    print_dict(params_init, indent=4)
    print_dict(params_opt, indent=4)
