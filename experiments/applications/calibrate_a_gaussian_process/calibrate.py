# todo: use the full time-series
# todo: use all output features
# todo: dense evaluation of the posterior (for plitting)
# todo: start with waaay worse parameters (so plots make sense)
# todo: make matrix-free


import functools

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


def plot_gp(
    ax,
    ms: gp.TimeSeriesData,
    ss: gp.TimeSeriesData,
    ds: gp.TimeSeriesData,
    warp=jnp.square,
):
    ax.plot(
        ds.inputs,
        warp(ds.targets),
        ".",
        markerfacecolor="orange",
        markeredgecolor="black",
    )
    plus = warp(ms.targets + 2 * ss.targets)
    minus = warp(ms.targets - 2 * ss.targets)
    ax.fill_between(
        ms.inputs.squeeze(), minus.squeeze(), plus.squeeze(), alpha=0.3, color="black"
    )
    ax.plot(ms.inputs, warp(ms.targets), color="black")
    ax.set_xlim((jnp.amin(ms.inputs), jnp.amax(ms.inputs)))
    ax.set_ylim((0.9 * jnp.amin(minus), 1.1 * jnp.amax(plus)))


# fetch dataset
inputs, targets = exp_util.uci_air_quality()
targets = jnp.sqrt(targets)

# Random states
key_ = jax.random.PRNGKey(1)
key_data, key_init = jax.random.split(key_, num=2)
noise_std = 0.1


num_pts = len(inputs) // 4
inputs, targets = 1.0 * inputs[:num_pts, None], targets[:num_pts]

kernel, params_like = gp.kernel_matern_32()
params = gp.parameters_init(key_init, params_like)
data = gp.TimeSeriesData(inputs, targets)


# Initial condition
means = gp.condition_mean(params, noise_std, kernel_fun=kernel, spatial_data=data)
stds = gp.condition_std(params, noise_std, kernel_fun=kernel, spatial_data=data)

# Start plotting
mosaic = [["before"], ["after"]]
fig, axes = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, dpi=200)
plot_gp(axes["before"], means[500:650], stds[500:650], data[500:650])
axes["before"].set_ylabel("Initial guess")

# Loss function
loss_p = functools.partial(
    gp.negative_log_likelihood, kernel_fun=kernel, spatial_data=data
)
loss = jax.jit(loss_p)

# Optimise
optim = jaxopt.BFGS(loss, verbose=True, maxiter=15)
result = optim.run((params, noise_std))
params_opt, noise_opt = result.params

# Print results
print("\nInitial guess:\n\t", noise_std, "\n\t", params.unravelled)
print("\nOptimised guess:\n\t", noise_opt, "\n\t", params_opt.unravelled)

# Plot results
means = gp.condition_mean(params_opt, noise_opt, kernel_fun=kernel, spatial_data=data)
stds = gp.condition_std(params_opt, noise_opt, kernel_fun=kernel, spatial_data=data)
plot_gp(axes["after"], means[500:650], stds[500:650], data[500:650])
axes["after"].set_ylabel("Optimized")

# Show results
plt.show()
