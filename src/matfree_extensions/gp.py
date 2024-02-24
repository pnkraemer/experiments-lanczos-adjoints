import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp


def kernel_matern_32():
    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            epsilon = jnp.finfo(x).eps
            diff = x - y
            scaled = scale_sqrt_in**2 * jnp.dot(diff, diff)
            sqrt = jnp.sqrt(scaled + epsilon)
            return scale_sqrt_out**2 * (1 + sqrt) * jnp.exp(-sqrt)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def kernel_quadratic_exponential():
    """Construct a square exponential kernel."""

    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            log_k = scale_sqrt_in**2 * jnp.dot(diff, diff)
            return scale_sqrt_out**2 * jnp.exp(-log_k)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def kernel_quadratic_rational():
    """Construct a rational quadratic kernel."""

    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            tmp = scale_sqrt_in**2 * jnp.dot(diff, diff)
            return scale_sqrt_out**2 / (1 + tmp)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def _vmap_gram(fun):
    tmp = jax.vmap(fun, in_axes=(0, None), out_axes=0)
    return jax.vmap(tmp, in_axes=(None, 1), out_axes=1)


@dataclasses.dataclass
class TimeSeriesData:
    inputs: jax.Array
    targets: jax.Array

    def __getitem__(self, item):
        return TimeSeriesData(self.inputs[item], self.targets[item])


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


# todo: data -> data
def condition_mean(parameters, noise_std, /, *, kernel_fun, data, inputs_eval):
    kernel_fun_p = kernel_fun(**parameters.unravelled)

    K = kernel_fun_p(data.inputs, data.inputs.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets)

    K_eval = kernel_fun_p(inputs_eval, data.inputs.T)
    mean = K_eval @ coeffs
    return TimeSeriesData(inputs_eval, mean)


def condition_std(parameters, noise_std, /, *, kernel_fun, data, inputs_eval):
    kernel_fun_p = kernel_fun(**parameters.unravelled)

    K = kernel_fun_p(data.inputs, data.inputs.T)
    eye = jnp.eye(len(K))

    K_xy = kernel_fun_p(inputs_eval, data.inputs.T)
    K_xx = kernel_fun_p(inputs_eval, inputs_eval.T)

    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, K_xy.T)
    stds = jnp.sqrt(jnp.diag(K_xx - K_xy @ coeffs))

    return TimeSeriesData(inputs_eval, stds)


def negative_log_likelihood(parameters_and_noise, /, *, kernel_fun, data):
    parameters, noise_std = parameters_and_noise
    kernel_fun_p = kernel_fun(**parameters.unravelled)
    K = kernel_fun_p(data.inputs, data.inputs.T)
    eye = jnp.eye(len(K))
    coeffs = jnp.linalg.solve(K + noise_std**2 * eye, data.targets)

    mahalanobis = data.targets @ coeffs
    _sign, entropy = jnp.linalg.slogdet(K + noise_std**2 * eye)
    return mahalanobis + entropy
