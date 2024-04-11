"""Gaussian process models."""

from typing import Callable

import jax
import jax.numpy as jnp

# todo: use matfree_extensions' Lanczos
from matfree import hutchinson, lanczos


def model(mean_fun: Callable, kernel_fun: Callable) -> Callable:
    """Construct a Gaussian process model."""

    def prior(x, **kernel_params):
        mean = mean_fun(x)
        cov = kernel_fun(**kernel_params)(x, x)
        return mean, cov

    return prior


def likelihood_gaussian() -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(mean, cov, *, raw_noise):
        # Apply a soft-plus because GPyTorch does
        return mean, cov + jnp.eye(len(cov)) * _softplus(raw_noise)

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def mll_exact(prior: Callable, likelihood: Callable, *, logpdf: Callable) -> Callable:
    """Construct a marginal log-likelihood function."""

    def mll(x, y, *params_logdet, params_prior: dict, params_likelihood: dict):
        # Evaluate the marginal data likelihood
        mean, cov = prior(x, **params_prior)
        mean_, cov_ = likelihood(mean, cov, **params_likelihood)

        # Evaluate the log-pdf
        value = logpdf(y, *params_logdet, mean=mean_, cov=cov_)

        # Normalise by the number of data points because GPyTorch does
        return value / len(x)

    return mll


def logpdf_scipy_stats():
    """Construct a logpdf function that wraps jax.scipy.stats."""

    def logpdf(y, /, *, mean, cov):
        return jax.scipy.stats.multivariate_normal.logpdf(y, mean=mean, cov=cov)

    return logpdf


def logpdf_cholesky():
    """Construct a logpdf function that relies on a Cholesky decomposition."""

    def logpdf(y, /, *, mean, cov):
        # Cholesky-decompose
        cholesky = jnp.linalg.cholesky(cov)

        # Log-determinant
        logdet = jnp.sum(jnp.log(jnp.diag(cholesky)))

        # Mahalanobis norm

        def solve_triangular(A, b):
            return jax.scipy.linalg.solve_triangular(A, b, lower=True, trans=False)

        tmp = solve_triangular(cholesky, y - mean)
        mahalanobis = jnp.dot(tmp, tmp)

        # Combine the terms
        n, _n = jnp.shape(cov)
        return -logdet - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi)

    return logpdf


def logpdf_lanczos(
    krylov_depth, /, *, slq_batch_num: int, slq_sample_num: int, slq_sample_type: str
):
    """Construct a logpdf function that relies on a Cholesky decomposition.

    If this logpdf is plugged into mll_exact(), the returned mll function
    evaluates as mll(x, y, key, params_prior=...)
    instead of mll(x, y, params_prior=...)

    The estimator uses slq_batch_num*slq_sample_num samples for SLQ.
    Use a single batch and increase slq_sample_num until memory limits occur.
    Then, increase the number of batches while keeping the batch size maximal.
    """

    match_sample_fun = {
        "rademacher": hutchinson.sampler_rademacher,
        "normal": hutchinson.sampler_normal,
    }

    def solve(A, b):
        result, _info = jax.scipy.sparse.linalg.cg(lambda s: A @ s, b)
        return result

    def logdet(A, key):
        # todo: use differentiable lanczos

        x_like = jnp.ones((len(A),), dtype=A.dtype)
        sampler = match_sample_fun[slq_sample_type](x_like, num=slq_sample_num)

        integrand = lanczos.integrand_spd(jnp.log, krylov_depth, lambda s, p: p @ s)

        estimate = hutchinson.hutchinson(integrand, sampler)

        keys = jax.random.split(key, num=slq_batch_num)
        values = jax.lax.map(lambda k: estimate(k, A), keys)
        return jnp.mean(values, axis=0) / 2

    def logpdf(y, *params_logdet, mean, cov):
        # Log-determinant
        logdet_ = logdet(cov, *params_logdet)

        # Mahalanobis norm
        tmp = solve(cov, y - mean)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        n, _n = jnp.shape(cov)
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

        return _vmap_gram(k)

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

        return _vmap_gram(k)

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

        return _vmap_gram(k)

    params_like = {
        "raw_lengthscale": jnp.empty(()),
        "raw_outputscale": jnp.empty(shape_out),
    }
    return parametrize, params_like


def _softplus(x, beta=1.0, threshold=20.0):
    # Shamelessly stolen from:
    # https://github.com/google/jax/issues/18443

    # mirroring the pytorch implementation https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
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


def _vmap_gram(fun):
    tmp = jax.vmap(fun, in_axes=(None, 0), out_axes=-1)
    return jax.vmap(tmp, in_axes=(0, None), out_axes=-2)
