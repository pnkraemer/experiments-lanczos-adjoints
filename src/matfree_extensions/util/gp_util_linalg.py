"""Linear algebra for Gaussian processes and Gram matrices."""

import warnings
from typing import Callable

import jax
import jax.numpy as jnp
import lineax
from matfree import hutchinson

from matfree_extensions import cg, lanczos


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


def krylov_solve_cg_jax(*, tol, maxiter):
    def solve(A: Callable, /, b: jax.Array):
        result, info = jax.scipy.sparse.linalg.cg(A, b, tol=tol, maxiter=maxiter)
        return result, info

    return solve


def krylov_solve_pcg_jax(*, tol, maxiter):
    def solve(A: Callable, /, b: jax.Array, P: Callable):
        result, info = jax.scipy.sparse.linalg.cg(A, b, tol=tol, maxiter=maxiter, M=P)
        return result, info

    return solve


def krylov_solve_cg_lineax(*, atol, rtol, max_steps):
    msg = "Lineax's CG is not differentiable. Use for debugging only."
    warnings.warn(msg, stacklevel=1)

    def solve(A: Callable, /, b: jax.Array):
        spd_tag = [lineax.symmetric_tag, lineax.positive_semidefinite_tag]
        op = lineax.FunctionLinearOperator(A, b, tags=spd_tag)
        solver = lineax.CG(atol=atol, rtol=rtol, max_steps=max_steps)
        solution = lineax.linear_solve(op, b, solver=solver)
        return solution.value, solution.stats

    return solve


def krylov_solve_pcg_lineax(*, atol, rtol, max_steps):
    msg = "Preconditioning with lineax is potentially broken."
    warnings.warn(msg, stacklevel=1)
    msg = "Lineax's CG is not differentiable. Use for debugging only."
    warnings.warn(msg, stacklevel=1)

    def solve(A: Callable, /, b: jax.Array, P: Callable):
        spd_tag = [lineax.symmetric_tag, lineax.positive_semidefinite_tag]
        op = lineax.FunctionLinearOperator(A, b, tags=spd_tag)
        solver = lineax.CG(atol=atol, rtol=rtol, max_steps=max_steps)

        precon = lineax.FunctionLinearOperator(P, b, tags=spd_tag)
        options = {"preconditioner": precon}
        solution = lineax.linear_solve(op, b, solver=solver, options=options)
        return solution.value, solution.stats

    return solve


def krylov_solve_cg_fixed_step(num_matvecs: int, /):
    return cg.cg_fixed_step(num_matvecs)


def krylov_solve_cg_fixed_step_reortho(num_matvecs: int, /):
    return cg.cg_fixed_step_reortho(num_matvecs)


def krylov_solve_pcg_fixed_step(num_matvecs: int, /):
    return cg.pcg_fixed_step(num_matvecs)


def krylov_solve_pcg_fixed_step_reortho(num_matvecs: int, /):
    return cg.pcg_fixed_step_reortho(num_matvecs)


def krylov_logdet_slq(
    krylov_depth, /, *, sample: Callable, num_batches: int, checkpoint: bool = False
):
    def logdet(A: Callable, /, key):
        integrand = lanczos.integrand_spd(jnp.log, krylov_depth, A)
        estimate = hutchinson.hutchinson(integrand, sample)

        # If a single batch, we never checkpoint.
        if num_batches == 1:
            return estimate(key)

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
