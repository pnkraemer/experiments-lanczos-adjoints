# todo: use more points
# todo: dense evaluation of the STD
# todo: why does the hyperparameter optimization not do much?
# todo: optimize noise
# todo: start with waaay worse parameters (so plots make sense)


import dataclasses
import functools
from typing import Callable

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


@dataclasses.dataclass
class TimeSeriesData:
    inputs: jax.Array
    targets: jax.Array

    def __getitem__(self, item):
        return TimeSeriesData(self.inputs[item], self.targets[item])


def data_plot(axis, d: TimeSeriesData, /, *, title="", **axis_kwargs):
    axis.set_title(title)
    axis.set_aspect("equal")
    return axis.contourf(*d.inputs, d.targets, **axis_kwargs)


@dataclasses.dataclass
class Params:
    ravelled: jax.Array
    unravel: Callable

    @property
    def unravelled(self):
        return self.unravel(self.ravelled)


def _flatten(p):
    return (p.ravelled,), (p.unravel,)


def _unflatten(a, c):
    return Params(*c, *a)


jax.tree_util.register_pytree_node(Params, _flatten, _unflatten)


def parameters_init(key, p, /):
    flat, unflatten = jax.flatten_util.ravel_pytree(p)
    flat_like = jax.random.normal(key, shape=flat.shape)
    return Params(flat_like, unflatten)


def condition_mean(parameters, /, *, kernel_fun, spatial_data):
    kernel_fun_p = kernel_fun(**parameters.unravelled)
    K = kernel_fun_p(spatial_data.inputs, spatial_data.inputs.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets)
    mean = K @ coeffs
    return TimeSeriesData(data.inputs, mean)


def condition_std(parameters, /, *, kernel_fun, spatial_data):
    kernel_fun_p = kernel_fun(**parameters.unravelled)
    K = kernel_fun_p(spatial_data.inputs, spatial_data.inputs.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, K)
    stds = jnp.sqrt(jnp.diag(K - K.T @ coeffs))
    return TimeSeriesData(data.inputs, stds)


def negative_log_likelihood(parameters, /, *, kernel_fun, spatial_data):
    kernel_fun_p = kernel_fun(**parameters.unravelled)
    K = kernel_fun_p(spatial_data.inputs, spatial_data.inputs.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets)

    mahalanobis = data.targets @ coeffs
    _sign, entropy = jnp.linalg.slogdet(K + noise_std**2 * eye)
    return mahalanobis + entropy


def plot_gp(ax, ms: TimeSeriesData, ss: TimeSeriesData, ds: TimeSeriesData):
    ax.plot(ds.inputs, jnp.exp(ds.targets), ".", color="orange")
    ax.fill_between(
        ms.inputs.squeeze(),
        jnp.exp(ms.targets - 3 * ss.targets).squeeze(),
        jnp.exp(ms.targets + 3 * ss.targets).squeeze(),
        alpha=0.3,
        color="black",
    )
    ax.plot(ms.inputs, jnp.exp(ms.targets), color="black")


# fetch dataset
inputs, targets = exp_util.uci_air_quality()
targets = jnp.log(targets)

# Random states
key_ = jax.random.PRNGKey(1)
key_data, key_init = jax.random.split(key_, num=2)
noise_std = 0.1


num_pts = len(inputs) // 4
inputs, targets = inputs[:num_pts, None], targets[:num_pts]

kernel, params_like = gp.kernel_quadratic_exponential()
params = parameters_init(key_init, params_like)
data = TimeSeriesData(inputs, targets)


# Initial condition
means = condition_mean(params, kernel_fun=kernel, spatial_data=data)
stds = condition_std(params, kernel_fun=kernel, spatial_data=data)

fig, axes = plt.subplot_mosaic([["before"], ["after"]], sharex=True, sharey=True)
plot_gp(axes["before"], means[500:1_000], stds[500:1_000], data[500:1_000])
axes["before"].set_ylabel("Initial guess")


loss_p = functools.partial(
    negative_log_likelihood, kernel_fun=kernel, spatial_data=data
)
loss = jax.jit(loss_p)
nll = loss(params)

optim = jaxopt.BFGS(loss, verbose=True, maxiter=5)
result = optim.run(params)
params_opt = result.params


print("\nInitial guess:\n\t", params.unravelled)
print("\nOptimised guess:\n\t", params_opt.unravelled)


means = condition_mean(params_opt, kernel_fun=kernel, spatial_data=data)
stds = condition_std(params_opt, kernel_fun=kernel, spatial_data=data)
plot_gp(axes["after"], means[500:1_000], stds[500:1_000], data[500:1_000])
axes["after"].set_ylabel("Optimized")

plt.show()
# plt.plot(data.inputs, jnp.exp(data.targets), "o", color="orange")
# plt.fill_between(
#     means.inputs.squeeze(),
#     jnp.exp(means.targets - 3 * stds.targets).squeeze(),
#     jnp.exp(means.targets + 3 * stds.targets).squeeze(),
#     alpha=0.3,
#     color="black",
# )
#
# plt.plot(means.inputs, jnp.exp(means.targets), color="black")
# plt.show()
