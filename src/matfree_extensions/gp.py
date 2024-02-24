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


def process_sample(key, *, noise_std, inputs, kernel, shape=()):
    """Sample a Gaussian process."""
    assert inputs.ndim == 2
    key_sample, key_noise = jax.random.split(key, num=2)

    K = kernel(inputs, inputs.T)
    U, s, Vt = jnp.linalg.svd(K)

    def transform(a, b):
        return U @ jnp.diag(jnp.sqrt(s)) @ Vt @ a + noise_std * b

    if shape == ():
        xi = jax.random.normal(key_sample, shape=(len(K),))
        eta = jax.random.normal(key_noise, shape=(len(K),))
        return transform(xi, eta)

    xi = jax.random.normal(key_sample, shape=(*shape, len(K)))
    eta = jax.random.normal(key_noise, shape=(*shape, len(K)))
    return jax.vmap(transform)(xi, eta)


def process_condition(inputs, targets, *, noise_std, kernel):
    """Condition a Gaussian process."""
    assert inputs.ndim == 2

    K = kernel(inputs, inputs.T)
    K += noise_std**2 * jnp.eye(len(K))
    coeff = jnp.linalg.solve(K, targets)

    def mean(x):
        assert x.ndim == 2
        return kernel(x, inputs.T) @ coeff

    def cov(x, y):
        assert x.ndim == 2
        assert y.ndim == 2
        k_xx = kernel(x, y.T)
        k_xy = jnp.linalg.solve(K, kernel(inputs, y.T))
        return k_xx - kernel(x, inputs.T) @ k_xy

    return mean, cov


# todo: use an actual logpdf function here.
def log_likelihood(inputs, targets, *, kernel, noise_std, solve_fun, slogdet_fun):
    """Evaluate the log-likelihood of observations."""
    assert inputs.ndim == 2
    assert targets.ndim == 1

    K = kernel(inputs, inputs.T)
    shift = noise_std**2 * jnp.eye(len(K))

    coeffs = solve_fun(K + shift, targets)
    residual_white = jnp.dot(targets, coeffs)

    _sign, logdet = slogdet_fun(K + shift)

    return -1 / 2 * (residual_white + logdet), (coeffs, jnp.linalg.cond(K + shift))
