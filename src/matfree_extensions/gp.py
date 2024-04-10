"""Gaussian process models."""

import jax
import jax.numpy as jnp


def kernel_periodic(shape_in, shape_out):
    def parametrize(*, scale_sqrt_in, scale_sqrt_out, period_sqrt):
        def k(x, y):
            _assert_shapes(x, y, shape_in)
            diff = x - y
            scaled = period_sqrt**2 * jnp.pi * jnp.sqrt(jnp.dot(diff, diff))

            inner_squared = scale_sqrt_in**2 * jnp.sin(scaled) ** 2
            return scale_sqrt_out**2 * jnp.exp(-inner_squared)

        return _vmap_gram(k)

    params_like = {
        "scale_sqrt_in": jnp.empty(()),
        "scale_sqrt_out": jnp.empty(shape_out),
        "period_sqrt": jnp.empty(shape_out),
    }
    return parametrize, params_like


def kernel_matern_32(*, shape_in, shape_out):
    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            diff = x - y
            scaled = scale_sqrt_in**2 * jnp.dot(diff, diff)

            # Shift by epsilon to guarantee differentiable sqrts
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return scale_sqrt_out**2 * (1 + sqrt) * jnp.exp(-sqrt)

        return _vmap_gram(k)

    params_like = {
        "scale_sqrt_in": jnp.empty(()),
        "scale_sqrt_out": jnp.empty(shape_out),
    }
    return parametrize, params_like


def kernel_matern_12(*, shape_in, shape_out):
    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            diff = x - y
            scaled = jnp.dot(diff, diff) / (scale_sqrt_in**2)

            # Shift by epsilon to guarantee differentiable sqrts
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return scale_sqrt_out**2 * jnp.exp(-sqrt)

        return _vmap_gram(k)

    params_like = {
        "scale_sqrt_in": jnp.empty(()),
        "scale_sqrt_out": jnp.empty(shape_out),
    }
    return parametrize, params_like


def kernel_quadratic_exponential(*, shape_in, shape_out):
    """Construct a square exponential kernel."""

    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            diff = x - y
            log_k = scale_sqrt_in**2 * jnp.dot(diff, diff)
            return scale_sqrt_out**2 * jnp.exp(-log_k)

        return _vmap_gram(k)

    params_like = {
        "scale_sqrt_in": jnp.empty(()),
        "scale_sqrt_out": jnp.empty(shape_out),
    }
    return parametrize, params_like


def kernel_quadratic_rational(*, shape_in, shape_out):
    """Construct a rational quadratic kernel."""

    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            diff = x - y
            tmp = scale_sqrt_in**2 * jnp.dot(diff, diff)
            return scale_sqrt_out**2 / (1 + tmp)

        return _vmap_gram(k)

    params_like = {
        "scale_sqrt_in": jnp.empty(()),
        "scale_sqrt_out": jnp.empty(shape_out),
    }
    return parametrize, params_like


def _assert_shapes(x, y, shape_in):
    if jnp.shape(x) != jnp.shape(y):
        error = "The arguments have different shapes: "
        error += f"{jnp.shape(x)} != {jnp.shape(y)})"
        raise ValueError(error)

    if jnp.shape(x) != shape_in:
        error = f"The shape {jnp.shape(x)} of the first argument "
        error += "does not match 'shape_in'={shape_in}"
        raise ValueError(error)


def _vmap_gram(fun):
    tmp = jax.vmap(fun, in_axes=(None, 1), out_axes=-1)
    return jax.vmap(tmp, in_axes=(0, None), out_axes=-2)
