"""Test the partial Cholesky decompositions."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions.util import gp_util_linalg


def case_low_rank_cholesky():
    """Construct a partial Cholesky factorisation."""
    return gp_util_linalg.low_rank_cholesky


def case_low_rank_cholesky_pivot():
    """Construct a partial Cholesky factorisation with pivoting."""
    return gp_util_linalg.low_rank_cholesky_pivot


@pytest_cases.parametrize_with_cases("low_rank", ".")
def test_full_rank_low_rank_cholesky_matches_full_cholesky(low_rank, n=5):
    key = jax.random.PRNGKey(2)

    cov_eig = 1.0 + jax.random.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation = low_rank(n, n)(lambda i, j: cov[i, j])

    tol = jnp.finfo(approximation.dtype).eps
    assert jnp.allclose(approximation @ approximation.T, cov, atol=tol, rtol=tol)


@pytest_cases.parametrize_with_cases("low_rank", ".")
def test_output_the_right_shapes(low_rank: Callable, n=4, rank=2):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation = low_rank(n, rank)(lambda i, j: cov[i, j])

    assert approximation.shape == (n, rank)


def test_pivoting_improves_the_estimate(n=10, rank=5):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    nopivot = gp_util_linalg.low_rank_cholesky(n, rank)(element)
    pivot = gp_util_linalg.low_rank_cholesky_pivot(n, rank)(element)

    error_nopivot = jnp.linalg.norm(cov - nopivot @ nopivot.T)
    error_pivot = jnp.linalg.norm(cov - pivot @ pivot.T)
    assert error_pivot < error_nopivot
