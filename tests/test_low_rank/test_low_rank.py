"""Test the partial Cholesky decompositions."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import low_rank


@pytest_cases.parametrize(
    "low_rank", [low_rank.cholesky_partial, low_rank.cholesky_partial_pivot]
)
def test_full_rank_cholesky_reconstructs_matrix(low_rank, n=5):
    key = jax.random.PRNGKey(2)

    cov_eig = 1.0 + jax.random.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = low_rank(rank=n)(lambda i, j: cov[i, j], n)

    tol = jnp.finfo(approximation.dtype).eps
    assert jnp.allclose(approximation @ approximation.T, cov, atol=tol, rtol=tol)


def test_full_rank_nopivot_matches_cholesky(n=10):
    key = jax.random.PRNGKey(2)
    cov_eig = 0.01 + jax.random.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    cholesky = jnp.linalg.cholesky(cov)

    # Sanity check: pivoting should definitely not satisfy this:
    received, info = low_rank.cholesky_partial_pivot(rank=n)(lambda i, j: cov[i, j], n)
    assert not jnp.allclose(received, cholesky)

    # But without pivoting, we should get there!
    received, info = low_rank.cholesky_partial(rank=n)(lambda i, j: cov[i, j], n)
    assert jnp.allclose(received, cholesky, atol=1e-6)


@pytest_cases.parametrize(
    "low_rank", [low_rank.cholesky_partial, low_rank.cholesky_partial_pivot]
)
def test_output_the_right_shapes(low_rank: Callable, n=4, rank=4):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, _info = low_rank(rank=rank)(lambda i, j: cov[i, j], n)
    assert approximation.shape == (n, rank)


def test_pivoting_improves_the_estimate(n=10, rank=5):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    nopivot, _info = low_rank.cholesky_partial(rank=rank)(element, n)
    pivot, _info = low_rank.cholesky_partial_pivot(rank=rank)(element, n)

    error_nopivot = jnp.linalg.norm(cov - nopivot @ nopivot.T)
    error_pivot = jnp.linalg.norm(cov - pivot @ pivot.T)
    assert error_pivot < error_nopivot


def test_preconditioner_solves_correctly(n=10):
    # Create a relatively ill-conditioned matrix
    cov_eig = 1.5 ** jnp.arange(-n // 2, n // 2, step=1.0)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    # Assert that the Cholesky decomposition is full-rank.
    cholesky = low_rank.cholesky_partial(rank=n)
    matrix, _info = cholesky(element, n)
    assert jnp.allclose(matrix @ matrix.T, cov)

    # Solve the linear system
    b = jnp.arange(1.0, 1 + len(cov))
    b /= jnp.linalg.norm(b)
    small_value = 1e-1
    cov_added = cov + small_value * jnp.eye(len(cov))

    expected = jnp.linalg.solve(cov_added, b)

    # Derive the preconditioner
    precondition = low_rank.preconditioner(cholesky)
    solve, info = precondition(element, n)

    # Test that the preconditioner solves correctly
    received = solve(b, small_value)
    assert jnp.allclose(received, expected, rtol=1e-2)
