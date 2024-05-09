"""Tests for selected logpdf functionality."""

import jax


def test_value_versus_cholesky(n=10):
    key = jax.random.PRNGKey(1)
    key1, key2 = jax.random.split(key, num=2)

    mean = jax.random.normal(key1, shape=(n,))
    cov = jax.random.normal(key2, shape=(n, n))
    cov = cov + cov.T + jnp.eye(n)

    print(mean, cov)
    assert False
