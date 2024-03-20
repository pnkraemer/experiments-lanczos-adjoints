# todo: use the full time-series
# todo: use all output features
# todo: make matrix-free
# todo: train-test split
# todo: reduce this down to a single quantity (NMLL/RMSE on test set?)
# todo: load and plot the input labels
# todo: plot samples instead of means

import functools
import json

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


def print_dict(dct, *, indent):
    dct = jax.tree_util.tree_map(float, dct)
    return json.dumps(dct, sort_keys=True, indent=indent)


def data_subsample(X, y, /, *, key, num):
    ints = jnp.arange(0, len(X))
    ints = jax.random.choice(key, ints, replace=False, shape=(num,))
    ints = jnp.sort(ints)
    return X[ints], y[ints]


def plot_gp(ax, ms: gp.TimeSeriesData, ss: gp.TimeSeriesData, ds: gp.TimeSeriesData):
    style_data = {
        "markerfacecolor": "orange",
        "markeredgecolor": "black",
        "linestyle": "None",
        "marker": ".",
    }
    ax.plot(ds.inputs.squeeze(), ds.targets.squeeze(), **style_data)

    style_mean = {"color": "black"}
    ax.plot(ms.inputs.squeeze(), ms.targets.squeeze(), **style_mean)

    style_std = {"color": "black", "alpha": 0.3}
    plus = ms.targets + 3 * ss.targets
    minus = ms.targets - 3 * ss.targets
    ax.fill_between(ms.inputs.squeeze(), minus.squeeze(), plus.squeeze(), **style_std)

    ax.set_xlim((jnp.amin(ms.inputs), jnp.amax(ms.inputs)))
    ax.set_ylim((-1 + jnp.amin(ds.targets), 1 + jnp.amax(ds.targets)))


# Initialise the random number generator
key_ = jax.random.PRNGKey(3)
key_data, key_init = jax.random.split(key_, num=2)

# Load and subsample the dataset
(inputs, targets) = exp_util.uci_air_quality()
inputs = inputs[..., None]  # (N, d) shape
num_pts = 20
inputs, targets = data_subsample(
    inputs[:num_pts], targets[:num_pts], key=key_data, num=num_pts
)

# Center the data
targets = jnp.log(targets)
bias = jnp.mean(targets)
targets = targets - bias


# Set up the model
kernel_matern_12, params_like_matern_12 = gp.kernel_matern_12()
kernel_matern_32, params_like_matern_32 = gp.kernel_matern_32()
kernel_periodic, params_like_periodic = gp.kernel_periodic()
params_like = {
    "matern_12": params_like_matern_12,
    "matern_32": params_like_matern_32,
    "periodic": params_like_periodic,
}


def kernel(*, matern_12, matern_32, periodic):
    def k(x, y):
        k_12 = kernel_matern_12(**matern_12)(x, y)
        k_32 = kernel_matern_32(**matern_32)(x, y)
        k_p = kernel_periodic(**periodic)(x, y)
        return k_12 + k_32 * k_p

    return k


params = gp.parameters_init(key_init, params_like)
data = gp.TimeSeriesData(inputs, targets)


# Plot the initial guess
noise_std = 1e-1 * jnp.ones(())
xlim = jnp.amin(inputs), jnp.amin(inputs) + 1.25 * (jnp.amax(inputs) - jnp.amin(inputs))
inputs_plot_1d = jnp.linspace(*xlim, num=200, endpoint=True)
inputs_plot = inputs_plot_1d[:, None]

gp_kwargs = {"kernel_fun": kernel, "data": data, "inputs_eval": inputs_plot}
means = gp.condition_mean(params, noise_std, **gp_kwargs)
stds = gp.condition_std(params, noise_std, **gp_kwargs)

# Start plotting
mosaic = [["before"], ["after"]]
fig, axes = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, dpi=200)
plot_gp(axes["before"], means, stds, data)
axes["before"].set_ylabel("Initial guess")

# Loss function
nmll = gp.negative_log_likelihood
loss_p = functools.partial(nmll, kernel_fun=kernel, data=data)
loss = jax.jit(loss_p)

# Optimise
optim = jaxopt.BFGS(loss, verbose=True, maxiter=1000)
result = optim.run((params, noise_std))
params_opt, noise_opt = result.params

# Print results
print(
    "\nInitial guess:\n\tnoise =",
    noise_std,
    "\n\tp =",
    print_dict(params.unravelled, indent=12),
)
print(
    "\nOptimised guess:\n\tnoise =",
    noise_opt,
    "\n\tp =",
    print_dict(params_opt.unravelled, indent=12),
)

# Plot results
print()

means = gp.condition_mean(params_opt, noise_opt, **gp_kwargs)
stds = gp.condition_std(params_opt, noise_opt, **gp_kwargs)
plot_gp(axes["after"], means, stds, data)
axes["after"].set_ylabel("Optimized")

# Show results
plt.show()
