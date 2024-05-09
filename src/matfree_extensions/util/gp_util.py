"""Gaussian process models."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp
from matfree import hutchinson

from matfree_extensions import lanczos

# todo: implement a logpdf with a custom vjp that reuses a CG call?!
#
# todo: if we implememt GP models as kernel(p, x),
#   and if model() expects a covariance_matvec function,
#   then we can unify the API with that in BNN_utils and we gain
#   a lot of functionality.


# todo: if we rename this to model_gp, we could
#  even place it next to model_mlp and whatnot
def model(mean_fun: Callable, kernel_fun: Callable, gram_matvec: Callable) -> Callable:
    """Construct a Gaussian process model."""

    def parametrise(**kernel_params):
        kfun = kernel_fun(**kernel_params)
        make_matvec = gram_matvec(kfun)

        def prior(x):
            mean = mean_fun(x)
            cov_matvec = functools.partial(make_matvec, x, x)
            return mean, cov_matvec

        return prior

    return parametrise


# todo: call this gram_matvec_sequential()?
def gram_matvec_map(*, checkpoint: bool):
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
def gram_matvec_map_over_batch(*, num_batches: int, checkpoint: bool):
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
        def matvec_map(x, y, v):
            num, *shape = jnp.shape(x)
            if num % num_batches != 0:
                msg = f"num_batches = {num_batches} does not divide dataset size {num}."
                raise ValueError(msg)

            mv = matvec_single(y, v)
            if checkpoint:
                mv = jax.checkpoint(mv)

            x_batched = jnp.reshape(x, (num_batches, num // num_batches, *shape))
            mapped = jax.lax.map(mv, x_batched)
            return jnp.reshape(mapped, (-1,))

        matvec_dense = gram_matvec_full_batch()
        matvec_dense_f = matvec_dense(fun)

        def matvec_single(y, v):
            def mv(x_batched):
                return matvec_dense_f(x_batched, y, v)

            return mv

        return matvec_map

    return matvec


# todo: call this gram_matvec()?
def gram_matvec_full_batch():
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product over a full batch.

    Use this function whenever the full Gram matrix
    fits into memory.
    Sometimes, on GPU, we get away with using this function
    even if the matrix does not fit (but usually not).
    """

    def matvec(fun: Callable) -> Callable:
        def matvec_y(x, y, v):
            fun_y = jax.vmap(fun, in_axes=(None, 0), out_axes=-1)
            return fun_y(x, y) @ v

        return jax.vmap(matvec_y, in_axes=(0, None, None), out_axes=0)

    return matvec


def gram_matrix(fun: Callable, /) -> Callable:
    """Turn a covariance function into a gram-matrix function."""
    tmp = jax.vmap(fun, in_axes=(None, 0), out_axes=-1)
    return jax.vmap(tmp, in_axes=(0, None), out_axes=-2)


# Todo: Ask for a shape input to have lengthscales per dimension?
def likelihood_gaussian() -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def parametrise(*, raw_noise):
        # Apply a soft-plus because GPyTorch does
        noise = _softplus(raw_noise)

        def likelihood(mean, cov):
            return mean, lambda v: cov(v) + noise * v

        return likelihood

    p = {"raw_noise": jnp.empty(())}
    return parametrise, p


def mll_exact(prior: Callable, likelihood: Callable, *, logpdf: Callable) -> Callable:
    """Construct a marginal log-likelihood function."""

    def mll(x, y, *params_logdet, params_prior: dict, params_likelihood: dict):
        # Evaluate the marginal data likelihood
        mean, cov = prior(**params_prior)(x)
        mean_, cov_ = likelihood(**params_likelihood)(mean, cov)

        # Evaluate the log-pdf
        value, info = logpdf(y, *params_logdet, mean=mean_, cov=cov_)

        # Normalise by the number of data points because GPyTorch does
        return value / len(x), info

    return mll


def logpdf_scipy_stats() -> Callable:
    """Construct a logpdf function that wraps jax.scipy.stats."""

    def logpdf(y, /, *, mean, cov: Callable):
        # Materialise the covariance matrix
        cov_matrix = jax.jacfwd(cov)(mean)

        _logpdf_fun = jax.scipy.stats.multivariate_normal.logpdf
        return _logpdf_fun(y, mean=mean, cov=cov_matrix), {}

    return logpdf


def logpdf_cholesky() -> Callable:
    """Construct a logpdf function that relies on a Cholesky decomposition."""

    def logpdf(y, /, *, mean, cov: Callable):
        # Materialise the covariance matrix
        cov_matrix = jax.jacfwd(cov)(mean)

        # Cholesky-decompose
        cholesky = jnp.linalg.cholesky(cov_matrix)

        # Log-determinant
        logdet = jnp.sum(jnp.log(jnp.diag(cholesky)))

        # Mahalanobis norm

        def solve_triangular(A, b):
            return jax.scipy.linalg.solve_triangular(A, b, lower=True, trans=False)

        tmp = solve_triangular(cholesky, y - mean)
        mahalanobis = jnp.dot(tmp, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)

        return -logdet - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), {}

    return logpdf


# todo: rename slq_* to mc_* to indicate monte carlo sampling?
def logpdf_lanczos(
    krylov_depth,
    /,
    slq_sampler: Callable,
    *,
    slq_batch_num: int,
    checkpoint: bool,
    cg_tol: float,
    cg_maxiter: int | None = None,  # same as in jax.scipy.sparse.cg
) -> Callable:
    """Construct a logpdf function that uses CG and Lanczos.

    Vocabulary:
        "CG" = "Conjugate gradient solver"
        "Lanczos" = The "Lanczos" iteration
        "SLQ" = "Stochastic Lanczos quadrature"
        (which combines Monte Carlo estimation with the Lanczos iteration)

    If this logpdf is plugged into mll_exact(), the returned mll function
    evaluates as mll(x, y, key, params_prior=...)
    instead of mll(x, y, params_prior=...)

    The estimator uses slq_batch_num*slq_sample_num samples for SLQ.
    Use a single batch and increase slq_sample_num until memory limits occur.
    Then, increase the number of batches while keeping the batch size maximal.

    Parameters
    ----------
    krylov_depth
        Depth of the Krylov space. Read this as:
        "Number of matrix-vector products" we want to use.
    slq_sampler
        Monte Carlo sampler for stochastic Lanczos quadrature.
        For instance, matfree.hutchinson.sampler_rademacher().
    slq_batch_num
        How many batches to use for Monte Carlo sampling
        This wraps around `slq_sampler`, so it multiplies
        however many samples slq_sampler generates.
        All batches are compute sequentially.
    checkpoint
        Whether to checkpoint each SLQ batch.
        Checkpoint increases memory efficiency but also runtime.
        Usually, we set this to `False`.
    cg_tol
        Tolerance to use for each conjugate gradients solve.
        For instance, GPyTorch uses `cg_tol=1.0` for
        Gaussian process hyperparameter calibration.
    cg_maxiter
        Maximum number of iterations for each conjugate
        gradients solve.
    """

    def solve(A: Callable, /, b):
        result, info = jax.scipy.sparse.linalg.cg(A, b, tol=cg_tol, maxiter=cg_maxiter)
        return result, info

    def logdet(A: Callable, /, key):
        integrand = lanczos.integrand_spd(jnp.log, krylov_depth, A)
        estimate = hutchinson.hutchinson(integrand, slq_sampler)

        # Memory-efficient reverse-mode derivatives
        #  See gram_matvec_map_over_batch().
        if checkpoint:
            estimate = jax.checkpoint(estimate)

        keys = jax.random.split(key, num=slq_batch_num)
        values = jax.lax.map(lambda k: estimate(k), keys)
        return jnp.mean(values, axis=0) / 2

    def logpdf(y, *params_logdet, mean, cov):
        # Log-determinant
        logdet_ = logdet(cov, *params_logdet)

        # Mahalanobis norm
        tmp, info = solve(cov, y - mean)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)
        return -logdet_ - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), info

    return logpdf


def logpdf_lanczos_reuse(
    krylov_depth, /, slq_sampler: Callable, slq_batch_num
) -> Callable:
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

    def solve(A: Callable, b):
        result, info = jax.scipy.sparse.linalg.cg(A, b)
        return result, info

    def logdet(A: Callable, key):
        integrand = lanczos.integrand_spd_custom_vjp_reuse(jnp.log, krylov_depth, A)
        estimate = hutchinson.hutchinson(integrand, slq_sampler)

        keys = jax.random.split(key, num=slq_batch_num)
        values = jax.lax.map(lambda k: estimate(k), keys)
        return jnp.mean(values, axis=0) / 2

    def logpdf(y, *params_logdet, mean, cov):
        # Log-determinant
        logdet_ = logdet(cov, *params_logdet)

        # Mahalanobis norm
        tmp, info = solve(cov, y - mean)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)
        return -logdet_ - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), info

    return logpdf


def mean_zero() -> Callable:
    """Construct a zero mean-function."""

    def mean(x):
        return jnp.zeros((len(x),), dtype=jnp.dtype(x))

    return mean


def kernel_scaled_matern_32(*, shape_in, shape_out) -> tuple[Callable, dict]:
    """Construct a (scaled) Matern(nu=3/2) kernel.

    The parametrisation equals that of GPyTorch's
    `ScaleKernel(MaternKernel(nu=1.5, constraint=Positive), constraint=Positive)`
    """

    def parametrize(*, raw_lengthscale, raw_outputscale):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            # Apply a soft-plus because GPyTorch does
            lengthscale = _softplus(raw_lengthscale)
            outputscale = _softplus(raw_outputscale)

            # Evaluate the scaled norm of differences
            # (expanding |x-y|^2 for GPU-speed)
            x = jnp.sqrt(3) * x / lengthscale
            y = jnp.sqrt(3) * y / lengthscale
            scaled = jnp.dot(x, x) + jnp.dot(y, y) - 2 * jnp.dot(x, y)

            # Clamp to allow square-roots
            scaled = jnp.maximum(0.0, scaled)

            # Shift by epsilon to guarantee differentiable sqrts
            # (Clamping is not enough because |x| is
            # not differentiable at zero
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return outputscale * (1 + sqrt) * jnp.exp(-sqrt)

        return k

    params_like = {
        "raw_lengthscale": jnp.empty(()),
        "raw_outputscale": jnp.empty(shape_out),
    }
    return parametrize, params_like


def kernel_scaled_matern_12(*, shape_in, shape_out) -> tuple[Callable, dict]:
    """Construct a (scaled) Matern(nu=1/2) kernel.

    The parametrisation equals that of GPyTorch's
    `ScaleKernel(MaternKernel(nu=0.5, constraint=Positive), constraint=Positive)`
    """

    def parametrize(*, raw_lengthscale, raw_outputscale):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            # Apply a soft-plus because GPyTorch does
            lengthscale = _softplus(raw_lengthscale)
            outputscale = _softplus(raw_outputscale)

            # Evaluate the scaled norm of differences
            # (expanding |x-y|^2 for GPU-speed)
            x /= lengthscale
            y /= lengthscale
            scaled = jnp.dot(x, x) + jnp.dot(y, y) - 2 * jnp.dot(x, y)

            # Clamp to allow square-roots
            scaled = jnp.maximum(0.0, scaled)

            # Shift by epsilon to guarantee differentiable sqrts
            # (Clamping is not enough because |x| is
            # not differentiable at zero
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return outputscale * jnp.exp(-sqrt)

        return k

    params_like = {
        "raw_lengthscale": jnp.empty(()),
        "raw_outputscale": jnp.empty(shape_out),
    }
    return parametrize, params_like


def kernel_scaled_rbf(*, shape_in, shape_out) -> tuple[Callable, dict]:
    """Construct a (scaled) radial basis function kernel.

    The parametrisation equals that of GPyTorch's
    `ScaleKernel(RBFKernel(constraint=Positive), constraint=Positive)`
    """

    def parametrize(*, raw_lengthscale, raw_outputscale):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            # Apply a soft-plus because GPyTorch does
            lengthscale = _softplus(raw_lengthscale)
            outputscale = _softplus(raw_outputscale)

            # Compute the norm of the differences:
            x /= lengthscale
            y /= lengthscale
            log_k = jnp.dot(x, x) + jnp.dot(y, y) - 2 * jnp.dot(x, y)

            # Clamp because the difference should never be negative
            log_k = jnp.maximum(0.0, log_k)

            # Return the kernel function
            return outputscale * jnp.exp(-log_k / 2)

        return k

    params_like = {
        "raw_lengthscale": jnp.empty(()),
        "raw_outputscale": jnp.empty(shape_out),
    }
    return parametrize, params_like


def _softplus(x, beta=1.0, threshold=20.0):
    # Shamelessly stolen from:
    # https://github.com/google/jax/issues/18443

    # mirroring the pytorch implementation
    #  https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    x_safe = jax.lax.select(x * beta < threshold, x, jax.numpy.ones_like(x))
    return jax.lax.select(
        x * beta < threshold,
        1 / beta * jax.numpy.log(1 + jax.numpy.exp(beta * x_safe)),
        x,
    )


def _assert_shapes(x, y, shape_in):
    if jnp.shape(x) != jnp.shape(y):
        error = "The arguments have different shapes: "
        error += f"{jnp.shape(x)} != {jnp.shape(y)})"
        raise ValueError(error)

    if jnp.shape(x) != shape_in:
        error = f"The shape {jnp.shape(x)} of the first argument "
        error += "does not match 'shape_in'={shape_in}"
        raise ValueError(error)


def cholesky_partial(matrix_element: Callable, n, rank: int):
    """Compute a partial Cholesky factorisation."""
    body = _cholesky_partial_makebody(matrix_element, n)
    # todo: handle parametrised matrix_element functions

    L = jnp.zeros((n, rank))
    return jax.lax.fori_loop(0, rank, body, L)


def _cholesky_partial_makebody(matrix_element: Callable, n: int):
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


def cholesky_partial_pivot(matrix_element: Callable, n, rank: int):
    """Compute a partial Cholesky factorisation with pivoting."""
    body = _cholesky_partial_pivot_body(matrix_element, n)

    # todo: handle parametrised matrix_element functions
    L = jnp.zeros((n, rank))
    P = jnp.arange(n)

    init = (L, P, P)
    (L, P, _matrix) = jax.lax.fori_loop(0, rank, body, init)
    return L, P


def _cholesky_partial_pivot_body(matrix_element: Callable, n: int):
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


def pivot_apply_inverse(arr, pivot, /):
    """Invert and apply a pivoting array to a matrix."""
    return arr[jnp.argsort(pivot)]
