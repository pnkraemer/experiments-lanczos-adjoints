import jax
import jax.numpy as jnp


def kernel_quadratic_exponential(*, gram_matrix: bool):
    def parametrize(*, scale_in, scale_out):
        def k(x, y):
            diff = x - y
            log_k = scale_in**2 * jnp.dot(diff, diff)
            return scale_out**2 * jnp.exp(log_k)

        if gram_matrix:
            return _vmap_gram(k)
        return k

    return parametrize


def kernel_quadratic_rational(*, gram_matrix: bool):
    def parametrize(*, scale_in, scale_out):
        def k(x, y):
            diff = x - y
            tmp = scale_in * jnp.dot(diff, diff)
            return scale_out / (1 + tmp)

        if gram_matrix:
            return _vmap_gram(k)
        return k

    return parametrize


def _vmap_gram(fun):
    tmp = jax.vmap(fun, in_axes=(0, None), out_axes=0)
    return jax.vmap(tmp, in_axes=(None, 1), out_axes=1)


def process_sample(key, *, noise, inputs, kernel, shape=()):
    assert inputs.ndim == 2
    key_sample, key_noise = jax.random.split(key, num=2)

    K = kernel(inputs, inputs.T)
    U, s, Vt = jnp.linalg.svd(K)

    def transform(a, b):
        return U @ jnp.diag(jnp.sqrt(s)) @ Vt @ a + noise * b

    if shape == ():
        xi = jax.random.normal(key_sample, shape=(len(K),))
        eta = jax.random.normal(key_noise, shape=(len(K),))
        return transform(xi, eta)

    xi = jax.random.normal(key_sample, shape=(*shape, len(K)))
    eta = jax.random.normal(key_noise, shape=(*shape, len(K)))
    return jax.vmap(transform)(xi, eta)


def process_condition(inputs, targets, *, noise, kernel):
    assert inputs.ndim == 2

    K = kernel(inputs, inputs.T)
    K += noise * jnp.eye(len(K))
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


def log_likelihood(inputs, targets, *, kernel, noise):
    assert inputs.ndim == 2
    assert targets.ndim == 1
    K = kernel(inputs, inputs.T)
    shift = noise * jnp.eye(len(K))

    coeffs = jnp.linalg.solve(K + shift, targets)
    mahalanobis = jnp.dot(targets, coeffs)
    _sign, entropy = jnp.linalg.slogdet(K)

    return -(mahalanobis + entropy), (coeffs, jnp.linalg.cond(K))
