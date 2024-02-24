# todo: make sure this module implements correct maths.
#  verify that the loss function is correct. The results are suspiciously bad.
# todo: reactivate the matfree components once the baseline is running...


import dataclasses
import functools
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from matfree_extensions import gp


@dataclasses.dataclass
class SpatialData:
    inputs_meshgrid: jax.Array
    targets_meshgrid: jax.Array

    @property
    def inputs_flat(self):
        d, *_ = jnp.shape(self.inputs_meshgrid)
        return jnp.reshape(self.inputs_meshgrid, (d, -1)).T

    @property
    def targets_flat(self):
        d, *_ = jnp.shape(self.inputs_meshgrid)
        return jnp.reshape(self.targets_meshgrid, (-1,))


def data_generate(key, inputs_meshgrid, targets_meshgrid, standard_deviation, /):
    noise = jax.random.normal(key, shape=targets_meshgrid.shape)
    noise_scaled = standard_deviation * noise
    return SpatialData(inputs_meshgrid, targets_meshgrid + noise_scaled)


def data_plot(axis, d: SpatialData, /, *, title="", **axis_kwargs):
    axis.set_title(title)
    return axis.contourf(*d.inputs_meshgrid, d.targets_meshgrid, **axis_kwargs)


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
    K = kernel_fun_p(spatial_data.inputs_flat, spatial_data.inputs_flat.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets_flat)
    mean = jnp.reshape(K @ coeffs, data.targets_meshgrid.shape)
    return SpatialData(data.inputs_meshgrid, mean)


def negative_log_likelihood(parameters, /, *, kernel_fun, spatial_data):
    kernel_fun_p = kernel_fun(**parameters.unravelled)
    K = kernel_fun_p(spatial_data.inputs_flat, spatial_data.inputs_flat.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets_flat)

    mahalanobis = data.targets_flat @ coeffs
    _sign, entropy = jnp.linalg.slogdet(K + noise_std**2 * eye)
    return mahalanobis + entropy


# Random states
key_ = jax.random.PRNGKey(1)
key_data, key_sol, key_slq, key_init = jax.random.split(key_, num=4)


# Figures
mosaic = [["post-before", "post-after", "truth"]]
fig_kwargs = {"dpi": 150, "sharex": True, "sharey": True, "figsize": (8, 2)}
fig, axes = plt.subplot_mosaic(mosaic, **fig_kwargs)


# Create data


def fun(x, y):
    return 1000 * ((x - 0.5) ** 2 + jnp.sin(y**2))


noise_std = 1e-1
xs_1d = jnp.linspace(0, 1, num=10, endpoint=True)
mesh_list = jnp.meshgrid(xs_1d, xs_1d)
mesh_in = jnp.stack(mesh_list)
mesh_out = fun(*mesh_in)

# todo: data_generate() should be add_noise()
truth = SpatialData(mesh_in, mesh_out)
data = data_generate(key_data, mesh_in, mesh_out, noise_std)

# Plot data
vmin = jnp.amin(data.targets_meshgrid)
vmax = jnp.amax(data.targets_meshgrid)
ax_kwargs = {"vmin": vmin, "vmax": vmax, "cmap": "plasma"}
clr = data_plot(axes["truth"], truth, title="Truth", **ax_kwargs)
fig.colorbar(clr)


# Create a GP
kernel, params_like = gp.kernel_quadratic_exponential()
params = parameters_init(key_init, params_like)

# Condition
evals = condition_mean(params, kernel_fun=kernel, spatial_data=data)


# Evaluate marginal log-likelihood
loss_p = functools.partial(
    negative_log_likelihood, kernel_fun=kernel, spatial_data=data
)
loss = jax.jit(loss_p)
nll = loss(params)

# Plot conditioning result (before optimization)
nll_init = loss(params)
title_init = f"Posterior (before; nmll={round(nll_init, 2):2F})"
data_plot(axes["post-before"], evals, title=title_init, **ax_kwargs)

# Optimize parameters
optim = jaxopt.LBFGS(loss, verbose=True)
result = optim.run(params)
params_opt = result.params

# Condition
evals = condition_mean(params_opt, kernel_fun=kernel, spatial_data=data)

# Plot conditioning result
nll_opt = loss(params_opt)
title_opt = f"Posterior (after; nmll={round(nll_opt, 2):2F})"
data_plot(axes["post-after"], evals, title=title_opt, **ax_kwargs)
plt.show()
