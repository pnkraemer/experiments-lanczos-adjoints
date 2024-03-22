import jax
import jax.numpy as jnp


def kernel_periodic():
    def parametrize(*, scale_sqrt_in, scale_sqrt_out, period_sqrt):
        def k(x, y):
            diff = x - y
            scaled = period_sqrt**2 * jnp.pi * jnp.sqrt(jnp.dot(diff, diff))

            inner_squared = scale_sqrt_in**2 * jnp.sin(scaled) ** 2
            return scale_sqrt_out**2 * jnp.exp(-inner_squared)

        return _vmap_gram(k)

    params_like = {
        "scale_sqrt_in": jnp.empty(()),
        "scale_sqrt_out": jnp.empty(()),
        "period_sqrt": jnp.empty(()),
    }
    return parametrize, params_like


def kernel_matern_32():
    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            scaled = scale_sqrt_in**2 * jnp.dot(diff, diff)

            # Shift by epsilon to guarantee differentiable sqrts
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return scale_sqrt_out**2 * (1 + sqrt) * jnp.exp(-sqrt)

        return _vmap_gram(k)

    empty = jnp.empty(())
    params_like = {"scale_sqrt_in": empty, "scale_sqrt_out": empty}
    return parametrize, params_like


def kernel_matern_12():
    def parametrize(*, scale_sqrt_in, scale_sqrt_out):
        def k(x, y):
            diff = x - y
            scaled = jnp.dot(diff, diff) / (scale_sqrt_in**2)

            # Shift by epsilon to guarantee differentiable sqrts
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return scale_sqrt_out**2 * jnp.exp(-sqrt)

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
