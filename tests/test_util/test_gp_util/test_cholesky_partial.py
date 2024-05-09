"""Fool around."""

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

    def cholesky_incomplete_pretend_pivoting(r):
        def decompose(x):
            alg = gp_util.cholesky_partial(r)
            return alg(x), jnp.arange(len(x))

        return decompose

    return cholesky_incomplete_pretend_pivoting


def case_cholesky_partial_pivot():
    """Construct a partial Cholesky factorisation with pivoting."""
    return gp_util.cholesky_partial_pivot


@pytest_cases.parametrize_with_cases("chol", ".")
def test_full_rank_partial_cholesky_matches_full_cholesky(chol, n=5):
    key = jax.random.PRNGKey(2)

    cov_eig = 1.0 + jax.random.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, piv = chol(n)(cov)
    approximation_p = gp_util.pivot_apply_inverse(approximation, piv)

    tol = jnp.finfo(approximation.dtype).eps
    assert jnp.allclose(approximation_p @ approximation_p.T, cov, atol=tol, rtol=tol)


@pytest_cases.parametrize_with_cases("chol", ".")
def test_output_the_right_shapes(chol: Callable, n=4, rank=2):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, piv = chol(rank)(cov)

    assert approximation.shape == (n, rank)
    assert piv.shape == (n,)


def test_pivoting_improves_the_estimate(n=10, rank=5):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    nopivot = gp_util.cholesky_partial(rank)(cov)
    pivot, p = gp_util.cholesky_partial_pivot(rank)(cov)
    pivot = gp_util.pivot_apply_inverse(pivot, p)

    error_nopivot = jnp.linalg.norm(cov - nopivot @ nopivot.T)
    error_pivot = jnp.linalg.norm(cov - pivot @ pivot.T)
    assert error_pivot < error_nopivot
