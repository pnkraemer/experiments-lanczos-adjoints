"""Test the partial Cholesky decompositions."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import low_rank


def case_low_rank_cholesky():
    """Construct a partial Cholesky factorisation."""
    return low_rank.cholesky_partial


def case_low_rank_cholesky_pivot():
    """Construct a partial Cholesky factorisation with pivoting."""
    return low_rank.cholesky_partial_pivot


@pytest_cases.parametrize_with_cases("low_rank", ".")
def test_full_rank_low_rank_cholesky_matches_full_cholesky(low_rank, n=5):
    key = jax.random.PRNGKey(2)

    cov_eig = 1.0 + jax.random.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = low_rank(n, n)(lambda i, j: cov[i, j])

    tol = jnp.finfo(approximation.dtype).eps
    assert jnp.allclose(approximation @ approximation.T, cov, atol=tol, rtol=tol)


@pytest_cases.parametrize_with_cases("low_rank", ".")
def test_output_the_right_shapes(low_rank: Callable, n=4, rank=2):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = low_rank(n, rank)(lambda i, j: cov[i, j])

    assert approximation.shape == (n, rank)


def test_pivoting_improves_the_estimate(n=10, rank=5):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    nopivot, _info = low_rank.cholesky_partial(n, rank)(element)
    pivot, _info = low_rank.cholesky_partial_pivot(n, rank)(element)

    error_nopivot = jnp.linalg.norm(cov - nopivot @ nopivot.T)
    error_pivot = jnp.linalg.norm(cov - pivot @ pivot.T)
    assert error_pivot < error_nopivot


def test_preconditioner_solves_correctly(n=10):
    # Create a relatively ill-conditioned matrix
    cov_eig = 2.0 ** jnp.arange(-n // 2, n // 2, step=1.0)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    # Assert that the Cholesky decomposition is full-rank.
    cholesky = low_rank.cholesky_partial_pivot(n, n)
    matrix, _info = cholesky(element)
    assert jnp.allclose(matrix @ matrix.T, cov)

    # Choose the small_value as well as possible
    eps = jnp.finfo(matrix.dtype).eps / jnp.linalg.cond(cov)
    small_value = jnp.sqrt(eps)

    # Derive the preconditioner
    precondition = low_rank.preconditioner(cholesky, small_value=small_value)
    solve, info = precondition(element)

    # Test that the preconditioner solves correctly
    b = jnp.arange(1.0, 1 + len(cov))
    received = solve(b)
    expected = jnp.linalg.solve(cov, b)
    assert jnp.allclose(received, expected, rtol=10 * jnp.sqrt(small_value))
