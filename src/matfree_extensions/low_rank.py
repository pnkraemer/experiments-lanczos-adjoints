"""Low-rank approximations."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp


def preconditioner(cholesky: Callable, /) -> Callable:
    """Turn a low-rank approximation into a preconditioner.

    Choose small_value small enough so that small_value * I + L L^\top \approx A
    holds, but large enough so that x \\mapsto x / small_value**2
    behaves well.

    A good guess is the square-root of machine precision.
    If the matrix is ill-conditioned, decrease the value.
    Some experimentation suggests that

    small_value = sqrt(machine_epsilon / cond(A))

    leads to decent results.
    """

    def solve_with_preconditioner(lazy_kernel, /, nrows: int):
        chol, info = cholesky(lazy_kernel, nrows)

        # Assert that the low-rank matrix is tall,
        # not wide (better safe than sorry)
        N, n = jnp.shape(chol)
        assert n <= N, (N, n)

        @jax.custom_vjp
        def solve(v: jax.Array, s: float):
            # Scale
            U = chol / jnp.sqrt(s)
            V = chol.T / jnp.sqrt(s)
            v /= s

            # Cholesky decompose the capacitance matrix
            # and solve the system
            eye_n = jnp.eye(n)
            chol_cap = jax.scipy.linalg.cho_factor(eye_n + V @ U)
            sol = jax.scipy.linalg.cho_solve(chol_cap, V @ v)
            return v - U @ sol

        # Ensure that no one ever differentiates through here

        def fwd(v, s):
            return solve(v, s), None

        def bwd(_cache, _vjp_incoming):
            raise RuntimeError

        solve.defvjp(fwd, bwd)

        return solve, info

    return solve_with_preconditioner


def cholesky_partial(*, rank: int) -> Callable:
    """Compute a partial Cholesky factorisation."""

    def cholesky(lazy_kernel: Callable, n: int, /):
        if rank > n:
            msg = f"Rank exceeds n: {rank} >= {n}."
            raise ValueError(msg)
        if rank < 1:
            msg = f"Rank must be positive, but {rank} < {1}."
            raise ValueError(msg)

        i, j = 0, 0
        element, aux_args = jax.closure_convert(lazy_kernel, i, j)
        return _cholesky(element, n, *aux_args)

    @functools.partial(jax.custom_vjp, nondiff_argnums=[0, 1])
    def _cholesky(lazy_kernel: Callable, n: int, *params):
        step = _cholesky_partial_body(lazy_kernel, n, *params)
        chol = jnp.zeros((n, rank))
        return jax.lax.fori_loop(0, rank, step, chol), {}

    # Ensure that no one ever differentiates through here

    def _fwd(*args):
        return _cholesky(*args), None

    def _bwd(*_args):
        raise RuntimeError

    _cholesky.defvjp(_fwd, _bwd)

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


def cholesky_partial_pivot(*, rank: int) -> Callable:
    """Compute a partial Cholesky factorisation with pivoting."""

    def cholesky(matrix_element: Callable, n: int):
        if rank > n:
            msg = f"Rank exceeds n: {rank} >= {n}."
            raise ValueError(msg)
        if rank < 1:
            msg = f"Rank must be positive, but {rank} < {1}."
            raise ValueError(msg)

        i, j = 0, 0
        element, aux_args = jax.closure_convert(matrix_element, i, j)
        return call_backend(element, n, *aux_args)

    @functools.partial(jax.custom_vjp, nondiff_argnums=[0])
    def call_backend(matrix_element: Callable, n: int, *params):
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
