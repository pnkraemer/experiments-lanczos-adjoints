"""Low-rank approximations."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp


def precondition(low_rank, small_value):
    """Turn a low-rank approximation into a preconditioner."""

    def precon(matrix, /):
        chol, info = low_rank(matrix)
        tmp = small_value * jnp.eye(len(chol.T)) + (chol.T @ chol)

        def matvec(v):
            tmp1 = v
            tmp2 = chol.T @ v
            return tmp1 - chol @ jnp.linalg.solve(tmp, tmp2) / small_value

        # Ensure that no one ever differentiates through here

        def matvec_fwd(v):
            return matvec(v), None

        def matvec_bwd(_cache, _vjp_incoming):
            raise RuntimeError

        matvec = jax.custom_vjp(matvec)
        matvec.defvjp(matvec_fwd, matvec_bwd)

        return matvec, info

    return precon


def cholesky_partial(n: int, rank: int):
    """Compute a partial Cholesky factorisation."""

    def cholesky(matrix_element: Callable):
        i, j = 0, 0
        element, aux_args = jax.closure_convert(matrix_element, i, j)
        return call_backend(element, *aux_args)

    @functools.partial(jax.custom_vjp, nondiff_argnums=[0])
    def call_backend(matrix_element: Callable, *params):
        body = _cholesky_partial_body(matrix_element, n, *params)

        L = jnp.zeros((n, rank))
        return jax.lax.fori_loop(0, rank, body, L), {}

    # Ensure that no one ever differentiates through here

    def fwd(*args):
        return call_backend(*args), None

    def bwd(*_args):
        raise RuntimeError

    call_backend.defvjp(fwd, bwd)

    return cholesky


def _cholesky_partial_body(fn: Callable, n: int, *args):
    idx = jnp.arange(n)

    def matrix_element(i, j):
        return fn(i, j, *args)

    def matrix_column(i):
        fun = jax.vmap(matrix_element, in_axes=(0, None))
        return fun(idx, i)

    def body(i, L):
        element = matrix_element(i, i)
        l_ii = jnp.sqrt(element - jnp.dot(L[i], L[i]))

        column = matrix_column(i)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii

        return L.at[:, i].set(l_ji)

    return body


def cholesky_partial_pivot(n: int, rank: int):
    """Compute a partial Cholesky factorisation with pivoting."""
    if rank > n:
        msg = f"Rank exceeds n: {rank} >= {n}."
        raise ValueError(msg)

    def cholesky(matrix_element: Callable):
        i, j = 0, 0
        element, aux_args = jax.closure_convert(matrix_element, i, j)
        return call_backend(element, *aux_args)

    @functools.partial(jax.custom_vjp, nondiff_argnums=[0])
    def call_backend(matrix_element: Callable, *params):
        body = _cholesky_partial_pivot_body(matrix_element, n, *params)

        L = jnp.zeros((n, rank))
        P = jnp.arange(n)

        init = (L, P, P, True)
        (L, P, _matrix, success) = jax.lax.fori_loop(0, rank, body, init)
        return _pivot_invert(L, P), {"success": success}

    # Ensure that no one ever differentiates through here

    def fwd(*args):
        return call_backend(*args), None

    def bwd(*_args):
        raise RuntimeError

    call_backend.defvjp(fwd, bwd)

    return cholesky


def _cholesky_partial_pivot_body(fn: Callable, n: int, *args):
    idx = jnp.arange(n)

    def matrix_element(i, j):
        return fn(i, j, *args)

    def matrix_element_p(i, j, *, permute):
        return matrix_element(permute[i], permute[j])

    def matrix_column_p(i, *, permute):
        fun = jax.vmap(matrix_element, in_axes=(0, None))
        return fun(permute[idx], permute[i])

    def matrix_diagonal_p(*, permute):
        fun = jax.vmap(matrix_element)
        return fun(permute[idx], permute[idx])

    def body(i, carry):
        L, P, P_matrix, success = carry

        # Access the matrix
        diagonal = matrix_diagonal_p(permute=P_matrix)

        # Find the largest entry for the residuals
        residual_diag = diagonal - jax.vmap(jnp.dot)(L, L)
        res = jnp.abs(residual_diag)
        k = jnp.argmax(res)

        # Pivot [pivot!!! pivot!!! pivot!!! :)]
        P_matrix = _swap_cols(P_matrix, i, k)
        L = _swap_rows(L, i, k)
        P = _swap_rows(P, i, k)

        # Access the matrix
        element = matrix_element_p(i, i, permute=P_matrix)
        column = matrix_column_p(i, permute=P_matrix)

        # Perform a Cholesky step
        # (The first line could also be accessed via
        #  residual_diag[k], but it might
        #  be more readable to do it again)
        l_ii_squared = element - jnp.dot(L[i], L[i])
        l_ii = jnp.sqrt(l_ii_squared)
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii
        success = jnp.logical_and(success, l_ii_squared > 0.0)

        # Update the estimate
        L = L.at[:, i].set(l_ji)
        return L, P, P_matrix, success

    return body


def _swap_cols(arr, i, j):
    return _swap_rows(arr.T, i, j).T


def _swap_rows(arr, i, j):
    ai, aj = arr[i], arr[j]
    arr = arr.at[i].set(aj)
    return arr.at[j].set(ai)


def _pivot_invert(arr, pivot, /):
    """Invert and apply a pivoting array to a matrix."""
    return arr[jnp.argsort(pivot)]
