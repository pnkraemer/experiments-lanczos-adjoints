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
    return json.dumps(dct, sort_keys=True, indent=indent)


def data_subsample(X, y, /, *, key, num):
    ints = jnp.arange(0, len(X))
    ints = jax.random.choice(key, ints, replace=False, shape=(num,))
    ints = jnp.sort(ints)
    return X[ints], y[ints]


#
# def plot_gp(
#     ax, ms: TimeSeriesData, ss: TimeSeriesData, ds: TimeSeriesData, samples=None
# ):
#     style_data = {
#         "markerfacecolor": "darkorange",
#         "markeredgecolor": "black",
#         "linestyle": "None",
#         "marker": ".",
#     }
#     ax.plot(ds.inputs.squeeze(), undo(ds.targets.squeeze()), **style_data)
#
#     style_mean = {"color": "black"}
#     ax.plot(ms.inputs.squeeze(), undo(ms.targets.squeeze()), **style_mean)
#
#     style_samples = {"linewidth": 0.5, "alpha": 0.8, "color": "darkorange"}
#     if samples is not None:
#         ax.plot(
#             samples.inputs.squeeze(), undo(samples.targets.squeeze()), **style_samples
#         )
#
#     style_std = {"color": "black", "alpha": 0.3}
#     plus = undo(ms.targets + 3 * ss.targets)
#     minus = undo(ms.targets - 3 * ss.targets)
#     ax.fill_between(ms.inputs.squeeze(), minus.squeeze(), plus.squeeze(), **style_std)
#
#     ax.set_xlim((jnp.amin(ms.inputs), jnp.amax(ms.inputs)))
#


def model():
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
        def k(x, y):
            k_12 = kernel_matern_12(**matern_12)(x, y)
            k_32 = kernel_matern_32(**matern_32)(x, y)
            k_p = kernel_periodic(**periodic)(x, y)
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
    num_pts = 10

    # Initialise the random number generator
    key_ = jax.random.PRNGKey(seed)
    key_data, key_init = jax.random.split(key_, num=2)

    # Load and subsample the dataset
    (inputs, targets) = exp_util.uci_air_quality()
    inputs = inputs[..., None]  # (N, d) shape
    inputs, targets = data_subsample(
        inputs[:num_pts], targets[:num_pts], key=key_data, num=num_pts
    )

    # Pre-process the data
    targets = jnp.log(targets)
    bias = jnp.mean(targets)
    scale = jnp.std(targets)
    targets = (targets - bias) / scale
    #
    # def undo(x):
    #     return jnp.exp(x * scale + bias)

    # Set up the model
    kernel, params_like = model()
    params_init = parameters_init(key_init, params_like)

    # Set up the loss function and optimize
    loss = functools.partial(
        negative_log_likelihood, kernel_fun=kernel, X=inputs, y=targets
    )
    optim = jaxopt.BFGS(loss, verbose=True, maxiter=100)
    result = optim.run(params_init)
    params_opt = result.params

    # Save the initial parameters
    directory = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory, exist_ok=True)

    parameters_save(params_init, directory, "params_init")
    parameters_save(params_opt, directory, "params_opt")

    print_dict(params_init, indent=4)
    print_dict(params_opt, indent=4)

    #
    # assert False
    # # Plot the initial guess
    # xlim = (
    #     jnp.amin(inputs),
    #     jnp.amin(inputs) + 1.25 * (jnp.amax(inputs) - jnp.amin(inputs)),
    # )
    # inputs_plot_1d = jnp.linspace(*xlim, num=1_000, endpoint=True)
    # inputs_plot = inputs_plot_1d[:, None]
    #
    # gp_kwargs = {"kernel_fun": kernel, "data": data, "inputs_eval": inputs_plot}
    # means_raw, covs = gp.condition(
    #     params.unravelled["kernel"], params.unravelled["noise"], **gp_kwargs
    # )
    # stds_raw = jnp.sqrt(jnp.diag(covs))
    # means = TimeSeriesData(inputs_plot, means_raw)
    # stds = TimeSeriesData(inputs_plot, stds_raw)
    #
    # # Start plotting
    # mosaic = [["before"], ["after"]]
    # fig, axes = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, dpi=200)
    # plot_gp(axes["before"], means, stds, data)
    # axes["before"].set_ylabel("Initial guess")
    #
    # # Loss function
    # nmll = gp.negative_log_likelihood
    # loss_p = functools.partial(nmll, kernel_fun=kernel, data=data)
    # loss = jax.jit(loss_p)
    #
    # # Optimise
    # optim = jaxopt.BFGS(loss, verbose=True, maxiter=100)
    # result = optim.run((params, noise_std))
    # params_opt, noise_opt = result.params
    #
    # # Print results
    # print(
    #     "\nInitial guess:\n\tnoise =",
    #     noise_std,
    #     "\n\tp =",
    #     print_dict(params.unravelled, indent=12),
    # )
    # print(
    #     "\nOptimised guess:\n\tnoise =",
    #     noise_opt,
    #     "\n\tp =",
    #     print_dict(params_opt.unravelled, indent=12),
    # )
    #
    # # Plot results
    # print()
    #
    # means_raw, covs = gp.condition(params_opt, noise_opt, **gp_kwargs)
    # stds_raw = jnp.sqrt(jnp.diag(covs))
    # means = TimeSeriesData(inputs_plot, means_raw)
    # stds = TimeSeriesData(inputs_plot, stds_raw)
    #
    # samples_base = jax.random.normal(key_, shape=(3, *means_raw.shape)).T
    # samples_raw = (
    #     means_raw[:, None]
    #     + jnp.linalg.cholesky(covs + jnp.eye(len(covs)) * 1e-4) @ samples_base
    # )
    # samples = TimeSeriesData(inputs_plot, samples_raw)
    # print(samples_raw)
    #
    # plot_gp(axes["after"], means, stds, data, samples=samples)
    # axes["after"].set_ylabel("Optimized")
    #
    # # Show results
    # plt.show()
