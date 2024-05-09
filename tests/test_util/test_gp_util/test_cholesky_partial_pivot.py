"""Fool around."""

import jax
import jax.numpy as jnp
from matfree import test_util


def cholesky_pivot_partial():
    def estimate(matrix):
        return matrix

    return estimate


def test_full_rank(n=4):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    reference = jnp.linalg.cholesky(cov)

    cholesky_p = cholesky_pivot_partial(n)
    approximation = cholesky_p(cov)
    assert jnp.allclose(approximation, reference)
