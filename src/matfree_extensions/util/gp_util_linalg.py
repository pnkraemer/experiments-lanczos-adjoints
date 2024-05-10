"""Linear algebra for Gaussian processes and Gram matrices."""

from typing import Callable

import jax
import jax.numpy as jnp
import lineax
from matfree import hutchinson

from matfree_extensions import lanczos


# todo: call this gram_matvec_sequential()?
def gram_matvec_map(*, checkpoint: bool = True):
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product by row-wise mapping.
    This function uses jax.lax.map.

    Use this function for gigantic matrices (on GPUs).

    Parameters
    ----------
    checkpoint
        Whether to wrap each row through jax.checkpoint.
        This increases both, memory efficiency and runtime.
        Setting it to `True` is generally a good idea.
    """

    def matvec(fun: Callable) -> Callable:
        def matvec_map(x, y, v):
            mv = matvec_single(y, v)
            if checkpoint:
                mv = jax.checkpoint(mv)

            mapped = jax.lax.map(mv, x)
            return jnp.reshape(mapped, (-1,))

        def matvec_single(y, v):
            def mv(x_single):
                return gram_matrix(fun)(x_single[None, ...], y) @ v

            return mv

        return matvec_map

    return matvec


# todo: call this gram_matvec_partitioned()?
# todo: rename num_batches to num_partitions?
def gram_matvec_map_over_batch(*, num_batches: int, checkpoint: bool = True):
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product by mapping over full batches.
    Why? To reduce memory compared to gram_matvec_full_batch,
    but to increase runtime compare to gram_matvec_map.

    Parameters
    ----------
    num_batches
        Number of batches. Make this value as large as possible,
        but small enough so that each batch fits into memory.
    checkpoint
        Whether to wrap each row through jax.checkpoint.
        This increases both, memory efficiency and runtime.
        Setting it to `True` is generally a good idea.

    Raises
    ------
    ValueError
        If the number of batches does not divide the dataset size.
        In this case, make them match by either subsampling data
        to, say, the nearest power of 2,
        or select a different number of batches.
    """

    def matvec(fun: Callable) -> Callable:
        def matvec_map(i, j, v):
            num, *shape = jnp.shape(i)
            if num % num_batches != 0:
                msg = f"num_batches = {num_batches} does not divide dataset size {num}."
                raise ValueError(msg)

            mv = matvec_single(j, v)
            if checkpoint:
                mv = jax.checkpoint(mv)

            x_batched = jnp.reshape(i, (num_batches, num // num_batches, *shape))
            mapped = jax.lax.map(mv, x_batched)
            return jnp.reshape(mapped, (-1,))

        matvec_dense = gram_matvec_full_batch()
        matvec_dense_f = matvec_dense(fun)

        def matvec_single(j, v):
            def mv(x_batched):
                return matvec_dense_f(x_batched, j, v)

            return mv

        return matvec_map

    return matvec


# todo: Rename to gram_matvec()?
def gram_matvec_full_batch():
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product over a full batch.

    Use this function whenever the full Gram matrix
    fits into memory.
    Sometimes, on GPU, we get away with using this function
    even if the matrix does not fit (but usually not).
    """

    def matvec(fun):
        def matvec_y(i, j, v: jax.Array):
            fun_j_batched = jax.vmap(fun, in_axes=(None, 0), out_axes=-1)
            return fun_j_batched(i, j) @ v

        return jax.vmap(matvec_y, in_axes=(0, None, None), out_axes=0)

    return matvec


def gram_matrix(fun: Callable, /) -> Callable:
    """Turn a covariance function into a gram-matrix function."""
    tmp = jax.vmap(fun, in_axes=(None, 0), out_axes=-1)
    return jax.vmap(tmp, in_axes=(0, None), out_axes=-2)


def precondition_low_rank(low_rank, small_value):
    """Turn a low-rank approximation into a preconditioner."""

    def precon(matrix, /):
        chol = low_rank(matrix)
        tmp = small_value * jnp.eye(len(chol.T)) + (chol.T @ chol)

        def matvec(v):
            tmp1 = v
            tmp2 = chol.T @ v
            return tmp1 - chol @ jnp.linalg.solve(tmp, tmp2) / small_value

        # Ensure that no one ever differentiates through here :)

        def matvec_fwd(v):
            return matvec(v), None

        def matvec_bwd(_cache, _vjp_incoming):
            raise RuntimeError

        matvec = jax.custom_vjp(matvec)
        matvec.defvjp(matvec_fwd, matvec_bwd)

        return matvec

    return precon


def low_rank_cholesky(n: int, rank: int):
    """Compute a partial Cholesky factorisation."""

    def call(matrix_element: Callable):
        body = _low_rank_cholesky_body(matrix_element, n)
        # todo: handle parametrised matrix_element functions

        L = jnp.zeros((n, rank))
        return jax.lax.fori_loop(0, rank, body, L)

    # # Todo: uncomment
    # # Ensure that no one ever differentiates through here :)
    #
    # def fwd(matrix_element):
    #     return call(matrix_element), None
    #
    # def bwd(_matrix_element, _cache, _vjp_incoming):
    #     raise RuntimeError
    #
    # call = jax.custom_vjp(call, nondiff_argnums=[0])
    # call.defvjp(fwd, bwd)

    return call


def _low_rank_cholesky_body(matrix_element: Callable, n: int):
    idx = jnp.arange(n)

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


def low_rank_cholesky_pivot(n: int, rank: int):
    """Compute a partial Cholesky factorisation with pivoting."""

    def call(matrix_element: Callable):
        body = _low_rank_cholesky_pivot_body(matrix_element, n)

        # todo: handle parametrised matrix_element functions
        L = jnp.zeros((n, rank))
        P = jnp.arange(n)

        init = (L, P, P)
        (L, P, _matrix) = jax.lax.fori_loop(0, rank, body, init)
        return _pivot_invert(L, P)

    # Todo: uncomment
    #
    # # Ensure that no one ever differentiates through here :)
    #
    # def fwd(matrix_element):
    #     return call(matrix_element), None
    #
    # def bwd(_matrix_element, _cache, _vjp_incoming):
    #     raise RuntimeError
    #
    # call = jax.custom_vjp(call, nondiff_argnums=[0])
    # call.defvjp(fwd, bwd)

    return call


def _low_rank_cholesky_pivot_body(matrix_element: Callable, n: int):
    idx = jnp.arange(n)

    def matrix_element_p(i, j, *, permute):
        return matrix_element(permute[i], permute[j])

    def matrix_column_p(i, *, permute):
        fun = jax.vmap(matrix_element, in_axes=(0, None))
        return fun(permute[idx], permute[i])

    def matrix_diagonal_p(*, permute):
        fun = jax.vmap(matrix_element)
        return fun(permute[idx], permute[idx])

    def body(i, carry):
        L, P, P_matrix = carry

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
        l_ii = jnp.sqrt(element - jnp.dot(L[i], L[i]))
        l_ji = column - L @ L[i, :]
        l_ji /= l_ii

        # Update the estimate
        L = L.at[:, i].set(l_ji)
        return L, P, P_matrix

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


def krylov_solve_cg(*, tol, maxiter):
    def solve(A: Callable, /, b: jax.Array):
        result, info = jax.scipy.sparse.linalg.cg(A, b, tol=tol, maxiter=maxiter)
        return result, info

    return solve


def krylov_solve_cg_precondition(*, tol, maxiter):
    def solve(A: Callable, /, b: jax.Array, P: Callable):
        result, info = jax.scipy.sparse.linalg.cg(A, b, tol=tol, maxiter=maxiter, M=P)
        return result, info

    return solve


def krylov_solve_cg_lineax(*, atol, rtol, max_steps):
    def solve(A: Callable, /, b: jax.Array):
        spd_tag = [lineax.symmetric_tag, lineax.positive_semidefinite_tag]
        op = lineax.FunctionLinearOperator(A, b, tags=spd_tag)
        solver = lineax.CG(atol=atol, rtol=rtol, max_steps=max_steps)
        solution = lineax.linear_solve(op, b, solver=solver)
        return solution.value, solution.stats

    return solve


def krylov_solve_cg_lineax_precondition(*, atol, rtol, max_steps):
    def solve(A: Callable, /, b: jax.Array, P: Callable):
        spd_tag = [lineax.symmetric_tag, lineax.positive_semidefinite_tag]
        op = lineax.FunctionLinearOperator(A, b, tags=spd_tag)
        solver = lineax.CG(atol=atol, rtol=rtol, max_steps=max_steps)

        precon = lineax.FunctionLinearOperator(P, b, tags=spd_tag)
        options = {"preconditioner": precon}
        solution = lineax.linear_solve(op, b, solver=solver, options=options)
        return solution.value, solution.stats

    return solve


def krylov_logdet_slq(
    krylov_depth, /, *, sample: Callable, num_batches: int, checkpoint: bool = False
):
    def logdet(A: Callable, /, key):
        integrand = lanczos.integrand_spd(jnp.log, krylov_depth, A)
        estimate = hutchinson.hutchinson(integrand, sample)

        # Memory-efficient reverse-mode derivatives
        #  See gram_matvec_map_over_batch().
        if checkpoint:
            estimate = jax.checkpoint(estimate)

        keys = jax.random.split(key, num=num_batches)
        values = jax.lax.map(lambda k: estimate(k), keys)
        return jnp.mean(values, axis=0)

    return logdet


def krylov_logdet_slq_vjp_reuse(
    krylov_depth, /, *, sample: Callable, num_batches: int, checkpoint: bool
):
    """Construct a logpdf function that uses CG and Lanczos for the forward pass.

    This method returns an algorithm similar to logpdf_lanczos_reuse;
    the only difference is that
    the gradient is estimated differently;
    while logpdf_lanczos() implements adjoints of Lanczos
    to get efficient *exact* gradients,
    logpdf_lanczos_reuse() recycles information from the forward
    pass to get extremely cheap, *inexact* gradients.

    This roughly relates to the following paper:

    @inproceedings{dong2017scalable,
        title={Scalable log determinants for {Gaussian} process kernel learning},
        author={Dong, Kun and
                Eriksson, David and
                Nickisch, Hannes and
                Bindel, David and
                Wilson, Andrew G},
        booktitle=NeurIPS,
        year={2017}
    }
    """

    def logdet(A: Callable, /, key):
        integrand = lanczos.integrand_spd_custom_vjp_reuse(jnp.log, krylov_depth, A)
        estimate = hutchinson.hutchinson(integrand, sample)

        # Memory-efficient reverse-mode derivatives
        #  See gram_matvec_map_over_batch().
        if checkpoint:
            estimate = jax.checkpoint(estimate)

        keys = jax.random.split(key, num=num_batches)
        values = jax.lax.map(lambda k: estimate(k), keys)
        return jnp.mean(values, axis=0)

    return logdet
