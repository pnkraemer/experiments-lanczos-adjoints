import jax.numpy as jnp


def kernel_exponentiated_quadratic():
    def parametrize(scale_in, scale_out):
        def k(x, y):
            diff = x - y
            log_k = scale_in * jnp.dot(diff, diff)
            return scale_out * jnp.exp(log_k)

        return k

    return parametrize


def kernel_rational_quadratic():
    def parametrize(scale_in, scale_out):
        def k(x, y):
            diff = x - y
            tmp = scale_in**2 * jnp.dot(diff, diff)
            return scale_out**2 / (1 + tmp)

        return k

    return parametrize
