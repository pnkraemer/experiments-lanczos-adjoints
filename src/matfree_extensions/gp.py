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


# todo: implement
#  gram_matvec_pmap_over_dense
#  gram_matvec_map_over_pmap
#  gram_matvec_map_over_pmap_over_dense


def gram_matvec_map():
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product by row-wise mapping.

    Use this function for gigantic matrices on GPUs.
    """

    def matvec(fun: Callable) -> Callable:
        def matvec_map(x, y, v):
            Kv_mapped = jax.lax.map(lambda x_: _matvec_single(x_, y, v), x)
            return jnp.reshape(Kv_mapped, (-1,))

        def _matvec_single(x_single, y, v):
            return gram_matrix(fun)(x_single[None, ...], y) @ v

        return matvec_map

    return matvec


def gram_matvec_map_over_batch(*, batch_size: int):
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product by mapping over full batches.

    This function is only useful for CPUs. Make the batch_size
    a divisor of the data set size, and choose the largest batch
    size such that `batch_size` rows of the Gram matrix fit into
    memory.
    """
    matvec_dense = gram_matvec_full_batch()

    def matvec(fun: Callable) -> Callable:
        def matvec_map(x, y, v):
            num, *shape = jnp.shape(x)
            if num % batch_size != 0:
                raise ValueError

            x_batched = jnp.reshape(x, (num // batch_size, batch_size, *shape))
            Kv_mapped = jax.lax.map(lambda x_: _matvec_single(x_, y, v), x_batched)
            return jnp.reshape(Kv_mapped, (-1,))

        matvec_dense_f = matvec_dense(fun)

        def _matvec_single(x_batched, y, v):
            return matvec_dense_f(x_batched, y, v)

        return matvec_map

    return matvec


def gram_matvec_full_batch():
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product over a full batch.

    On CPU, use this function whenever the full Gram matrix
    fits into memory.

    On GPU, always use this function as the preferred method
    (it seems to work even for gigantic matrices).
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


def likelihood_gaussian() -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def parametrise(*, raw_noise):
        noise = _softplus(raw_noise)

        def likelihood(mean, cov):
            # Apply a soft-plus because GPyTorch does
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
        value = logpdf(y, *params_logdet, mean=mean_, cov=cov_)

        # Normalise by the number of data points because GPyTorch does
        return value / len(x)

    return mll


def logpdf_scipy_stats() -> Callable:
    """Construct a logpdf function that wraps jax.scipy.stats."""

    def logpdf(y, /, *, mean, cov: Callable):
        # Materialise the covariance matrix
        cov_matrix = jax.jacfwd(cov)(mean)

        _logpdf_fun = jax.scipy.stats.multivariate_normal.logpdf
        return _logpdf_fun(y, mean=mean, cov=cov_matrix)

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
        return -logdet - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi)

    return logpdf


def logpdf_lanczos(krylov_depth, /, slq_sampler: Callable, slq_batch_num) -> Callable:
    """Construct a logpdf function that uses CG and Lanczos.

    If this logpdf is plugged into mll_exact(), the returned mll function
    evaluates as mll(x, y, key, params_prior=...)
    instead of mll(x, y, params_prior=...)

    The estimator uses slq_batch_num*slq_sample_num samples for SLQ.
    Use a single batch and increase slq_sample_num until memory limits occur.
    Then, increase the number of batches while keeping the batch size maximal.
    """

    def solve(A: Callable, /, b):
        result, _info = jax.scipy.sparse.linalg.cg(A, b)
        return result

    def logdet(A: Callable, /, key):
        integrand = lanczos.integrand_spd(jnp.log, krylov_depth, A)
        estimate = hutchinson.hutchinson(integrand, slq_sampler)

        keys = jax.random.split(key, num=slq_batch_num)
        values = jax.lax.map(lambda k: estimate(k), keys)
        return jnp.mean(values, axis=0) / 2

    def logpdf(y, *params_logdet, mean, cov):
        # Log-determinant
        logdet_ = logdet(cov, *params_logdet)

        # Mahalanobis norm
        tmp = solve(cov, y - mean)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)
        return -logdet_ - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi)

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
        result, _info = jax.scipy.sparse.linalg.cg(A, b)
        return result

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
        tmp = solve(cov, y - mean)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)
        return -logdet_ - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi)

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

            diff = jnp.sqrt(3) * (x - y) / lengthscale
            scaled = jnp.dot(diff, diff)

            # Shift by epsilon to guarantee differentiable sqrts
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

            # Compute the norm of the differences
            diff = (x - y) / lengthscale
            scaled = jnp.dot(diff, diff)

            # Shift by epsilon to guarantee differentiable sqrts
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

            # Compute the norm of the differences
            diff = (x - y) / lengthscale
            log_k = jnp.dot(diff, diff)

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
