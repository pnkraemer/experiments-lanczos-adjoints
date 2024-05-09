"""Fool around."""

import jax
import jax.numpy as jnp
from matfree import test_util


def cholesky_pivot_partial(rank):
    def estimate(matrix):
        return matrix[:, :rank]

    return estimate


def test_full(n=4):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    reference = jnp.linalg.cholesky(cov)

    cholesky_p = cholesky_pivot_partial(n)
    approximation = cholesky_p(cov)
    assert jnp.allclose(approximation, reference)


def test_partial(n=4, rank=2):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    cholesky_p = cholesky_pivot_partial(rank)
    approximation = cholesky_p(cov)

    assert approximation.shape == (n, rank)
