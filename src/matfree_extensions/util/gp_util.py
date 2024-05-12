"""Gaussian process models."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp

# todo: implement a logpdf with a custom vjp that reuses a CG call?!


# todo: if we rename this to model_gp, we could
#  even place it next to model_mlp and whatnot
def model(mean_fun: Callable, kernel_fun: Callable) -> Callable:
    """Construct a Gaussian process model."""

    def prior(x: jax.Array, params: dict):
        # todo: what exactly is this function doing?

        kernel = kernel_fun(**params)
        return mean_fun, (kernel, x)

    return prior


def likelihood_gaussian_pdf(
    gram_matvec: Callable, logpdf: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(mean, covariance, params: dict):
        # Apply a soft-plus because GPyTorch does
        raw_noise = params["raw_noise"]
        noise = _softplus(raw_noise)

        kernel, inputs = covariance

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = gram_matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        def logpdf_partial(targets, *p_logpdf):
            return logpdf(targets, *p_logpdf, mean=mean(inputs), cov_matvec=cov_matvec)

        return logpdf_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def likelihood_gaussian_pdf_p(
    gram_matvec: Callable, logpdf_p: Callable, precondition: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(mean, covariance, params: dict):
        # Apply a soft-plus because GPyTorch does
        raw_noise = params["raw_noise"]
        noise = _softplus(raw_noise)

        kernel, inputs = covariance

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = gram_matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        pre, info = precondition(lazy_kernel, len(inputs))

        def logpdf_partial(targets, *p_logpdf):
            val, aux = logpdf_p(
                targets,
                *p_logpdf,
                mean=mean(inputs),
                cov_matvec=cov_matvec,
                P=lambda v: pre(v, noise),
            )
            return val, {"precondition": info, "logpdf": aux}

        return logpdf_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def likelihood_gaussian_condition(
    gram_matvec: Callable, solve: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(mean, covariance, params: dict):
        # Apply a soft-plus because GPyTorch does
        raw_noise = params["raw_noise"]
        noise = _softplus(raw_noise)

        kernel, inputs = covariance

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = gram_matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        def condition_partial(xs, targets):
            weights, info = solve(cov_matvec, targets - mean(inputs))

            def cov_matvec_prior(v):
                cov = gram_matvec(kernel)
                return cov(xs, inputs, v)

            return mean(xs) + cov_matvec_prior(weights), {"solve": info}

        return condition_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


# todo: if we build the preconditioner internally,
#  GP-model constructors simplify a lot
def mll_exact(prior: Callable, likelihood_pdf: Callable) -> Callable:
    """Construct a marginal log-likelihood function."""

    def mll(inputs, targets, *p_logpdf, params_prior: dict, params_likelihood: dict):
        # Evaluate the marginal data likelihood
        mean, kernel = prior(inputs, params=params_prior)
        loss = likelihood_pdf(mean, kernel, params=params_likelihood)
        value, info_pdf = loss(targets, *p_logpdf)

        # Normalise by the number of data points because GPyTorch does
        return value / len(inputs), info_pdf

    return mll


def posterior_exact(prior: Callable, likelihood: Callable) -> Callable:
    """Construct a marginal log-likelihood function."""

    def posterior(inputs, targets, params_prior: dict, params_likelihood: dict):
        # Evaluate the marginal data likelihood
        mean, kernel = prior(inputs, params=params_prior)
        condition = likelihood(mean, kernel, params=params_likelihood)
        return functools.partial(condition, targets=targets), {}

    return posterior


def logpdf_scipy_stats() -> Callable:
    """Construct a logpdf function that wraps jax.scipy.stats."""

    def logpdf(y, /, *, mean, cov_matvec: Callable):
        # Materialise the covariance matrix
        cov_matrix = jax.jacfwd(cov_matvec)(mean)

        _logpdf_fun = jax.scipy.stats.multivariate_normal.logpdf
        return _logpdf_fun(y, mean=mean, cov=cov_matrix), {}

    return logpdf


def logpdf_cholesky() -> Callable:
    """Construct a logpdf function that relies on a Cholesky decomposition."""

    def logpdf(y, /, *, mean, cov_matvec: Callable):
        # Materialise the covariance matrix
        cov_matrix = jax.jacfwd(cov_matvec)(mean)

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


def logpdf_krylov(solve: Callable, logdet: Callable):
    def logpdf(y, *params_logdet, mean, cov_matvec: Callable):
        # Log-determinant
        logdet_ = logdet(cov_matvec, *params_logdet) / 2

        # Mahalanobis norm
        tmp, info = solve(cov_matvec, y - mean)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)
        return -logdet_ - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), info

    return logpdf


def logpdf_krylov_p(solve_p: Callable, logdet: Callable):
    """Evaluate a logpdf via preconditioned Krylov methods."""

    def logpdf(y, *params_logdet, mean, cov_matvec, P: Callable):
        # Log-determinant
        logdet_ = logdet(cov_matvec, *params_logdet) / 2

        # Mahalanobis norm
        tmp, info = solve_p(cov_matvec, y - mean, P=P)
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
            # not differentiable at zero)
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return outputscale * (1 + sqrt) * jnp.exp(-sqrt)

        return k

    params_like = {
        "raw_lengthscale": jnp.empty(shape_in),
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
            # not differentiable at zero)
            epsilon = jnp.finfo(scaled).eps
            sqrt = jnp.sqrt(scaled + epsilon)
            return outputscale * jnp.exp(-sqrt)

        return k

    params_like = {
        "raw_lengthscale": jnp.empty(shape_in),
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
        "raw_lengthscale": jnp.empty(shape_in),
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
