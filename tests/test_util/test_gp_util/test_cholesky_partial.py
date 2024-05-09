"""Fool around."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util


def case_partial():
    def cholesky_partial(rank):
        def estimate(matrix):
            L = jnp.zeros((len(matrix), rank))

            body = makebody(matrix)
            return jax.lax.fori_loop(0, rank, body, L)

        def makebody(matrix):
            def body(i, L):
                l_ii = jnp.sqrt(matrix[i, i] - jnp.dot(L[i], L[i]))

                l_ji = matrix[:, i] - L @ L[i, :]
                l_ji /= l_ii

                return L.at[:, i].set(l_ji)

            return body

        return estimate

    return cholesky_partial


@pytest_cases.parametrize_with_cases("chol", ".")
def test_full(chol: Callable, n=5):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    reference = jnp.linalg.cholesky(cov)

    approximation = chol(n)(cov)

    tol = jnp.finfo(approximation.dtype).eps
    assert jnp.allclose(approximation, reference, atol=tol, rtol=tol)


@pytest_cases.parametrize_with_cases("chol", ".")
def test_partial(chol: Callable, n=4, rank=2):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation = chol(rank)(cov)

    assert approximation.shape == (n, rank)
