"""Fool around."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util


def case_cholesky_partial():
    """Construct a partial Cholesky factorisation.

    Return a function whose API looks like that of cholesky_partial_pivot,
    so we only write a single test suite.
    """

    def cholesky_incomplete_pretend_pivoting(r):
        def decompose(x):
            alg = cholesky_partial(r)
            return alg(x), jnp.arange(len(x))

        return decompose

    return cholesky_incomplete_pretend_pivoting


def case_cholesky_partial_pivot():
    """Construct a partial Cholesky factorisation with pivoting."""
    return cholesky_partial_pivot


def cholesky_partial(rank):
    """Construct a partial Cholesky factorisation."""

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


def cholesky_partial_pivot(rank):
    """Construct a partial Cholesky factorisation with pivoting."""

    def estimate(matrix):
        L = jnp.zeros((len(matrix), rank))
        P = jnp.arange(len(matrix))
        for i in range(rank):
            # Find the largest entry for the residuals
            residual = matrix - L @ L.T
            res = jnp.abs(jnp.diag(residual))
            k = jnp.argmax(res)

            # Pivot (pivot!!! pivot!!! pivot!!!)
            matrix = _swap_rows(matrix, i, k)
            matrix = _swap_cols(matrix, i, k)
            L = _swap_rows(L, i, k)
            P = _swap_rows(P, i, k)

            # Perform a Cholesky step
            l_ii = jnp.sqrt(matrix[i, i] - jnp.dot(L[i], L[i]))
            l_ji = matrix[:, i] - L @ L[i, :]
            l_ji /= l_ii

            # Update the estimate
            L = L.at[:, i].set(l_ji)

        return L, P

    return estimate


def _swap_cols(arr, i, j):
    return _swap_rows(arr.T, i, j).T


def _swap_rows(arr, i, j):
    ai, aj = arr[i], arr[j]
    arr = arr.at[i].set(aj)
    return arr.at[j].set(ai)


def pivot_apply_inverse(arr, pivot, /):
    """Invert and apply a pivoting array to a matrix."""
    return arr[jnp.argsort(pivot)]


@pytest_cases.parametrize_with_cases("chol", ".")
def test_full_rank_partial_cholesky_matches_full_cholesky(chol, n=5):
    key = jax.random.PRNGKey(2)

    cov_eig = 1.0 + jax.random.uniform(key, shape=(n,), dtype=float)
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)

    approximation, piv = chol(n)(cov)
    approximation_p = pivot_apply_inverse(approximation, piv)

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

    nopivot = cholesky_partial(rank)(cov)
    pivot, p = cholesky_partial_pivot(rank)(cov)
    pivot = pivot_apply_inverse(pivot, p)

    error_nopivot = jnp.linalg.norm(cov - nopivot @ nopivot.T)
    error_pivot = jnp.linalg.norm(cov - pivot @ pivot.T)
    assert error_pivot < error_nopivot
