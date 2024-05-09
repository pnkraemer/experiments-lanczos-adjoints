"""Test the partial Cholesky decompositions."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions.util import gp_util


def case_cholesky_partial():
    """Construct a partial Cholesky factorisation.

    Return a function whose API looks like that of cholesky_partial_pivot,
    so we only write a single test suite.
    """

    def cholesky_incomplete_pretend_pivoting(fun, n, rank):
        x = gp_util.cholesky_partial(fun, n, rank)
        return x, jnp.arange(n)

    return cholesky_incomplete_pretend_pivoting


def case_cholesky_partial_pivot():
    """Construct a partial Cholesky factorisation with pivoting."""
    return gp_util.cholesky_partial_pivot


@pytest_cases.parametrize_with_cases("cholesky", ".")
def test_full_rank_partial_cholesky_matches_full_cholesky(cholesky, n=5):
    key = jax.random.PRNGKey(2)

    cov_eig = 1.0 + jax.random.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, piv = cholesky(lambda i, j: cov[i, j], n, n)
    approximation_p = gp_util.pivot_apply_inverse(approximation, piv)

    tol = jnp.finfo(approximation.dtype).eps
    assert jnp.allclose(approximation_p @ approximation_p.T, cov, atol=tol, rtol=tol)


@pytest_cases.parametrize_with_cases("cholesky", ".")
def test_output_the_right_shapes(cholesky: Callable, n=4, rank=2):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, piv = cholesky(lambda i, j: cov[i, j], n, rank)

    assert approximation.shape == (n, rank)
    assert piv.shape == (n,)


def test_pivoting_improves_the_estimate(n=10, rank=5):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    def element(i, j):
        return cov[i, j]

    nopivot = gp_util.cholesky_partial(element, n, rank)
    pivot, p = gp_util.cholesky_partial_pivot(element, n, rank)
    pivot = gp_util.pivot_apply_inverse(pivot, p)

    error_nopivot = jnp.linalg.norm(cov - nopivot @ nopivot.T)
    error_pivot = jnp.linalg.norm(cov - pivot @ pivot.T)
    assert error_pivot < error_nopivot
