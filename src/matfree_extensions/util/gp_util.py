"""Gaussian process models."""

from typing import Callable

import jax
import jax.numpy as jnp

# todo: implement a logpdf with a custom vjp that reuses a CG call?!


# todo: if we rename this to model_gp, we could
#  even place it next to model_mlp and whatnot
def model(mean_fun: Callable, kernel_fun: Callable) -> Callable:
    """Construct a Gaussian process model."""

    def prior(x: jax.Array, params: dict):
        mean = mean_fun(x)
        kernel = kernel_fun(**params)

        def lazy_kernel(i: int, j: int) -> jax.Array:
            return kernel(x[i], x[j])

        return mean, lazy_kernel

    return prior


# Todo: Ask for a shape input to have lengthscales per dimension?
def likelihood_gaussian() -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(mean: jax.Array, lazy_kernel_prior: Callable, params: dict):
        # Apply a soft-plus because GPyTorch does
        raw_noise = params["raw_noise"]
        noise = _softplus(raw_noise)

        def lazy_kernel(i: int, j: int):
            return lazy_kernel_prior(i, j) + noise * (i == j)

        return mean, lazy_kernel

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


# todo: rename to lml,
#  because it is a log-marginal-likelihood, not a marginal-log-likelihood
def mll_exact(
    prior: Callable, likelihood: Callable, *, logpdf: Callable, gram_matvec: Callable
) -> Callable:
    """Construct a marginal log-likelihood function."""

    def mll(inputs, targets, *p_logpdf, params_prior: dict, params_likelihood: dict):
        # Evaluate the marginal data likelihood
        mean, kernel = prior(inputs, params=params_prior)
        mean_, kernel_ = likelihood(mean, kernel, params=params_likelihood)

        # Build matvec

        def cov_matvec(v):
            cov = gram_matvec(kernel_)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        # Evaluate the log-pdf
        value, info = logpdf(targets, *p_logpdf, mean=mean_, cov_matvec=cov_matvec)

        # Normalise by the number of data points because GPyTorch does
        return value / len(inputs), {"logpdf": info}

    return mll


def mll_exact_p(
    prior: Callable,
    likelihood: Callable,
    *,
    logpdf_p: Callable,
    gram_matvec: Callable,
    precondition: Callable,
) -> Callable:
    """Construct a marginal log-likelihood function."""

    def mll(inputs, targets, *p_logpdf, params_prior: dict, params_likelihood: dict):
        # Evaluate the marginal data likelihood
        mean, kernel = prior(inputs, params=params_prior)
        mean_, kernel_ = likelihood(mean, kernel, params=params_likelihood)

        # Build matvec
        cov = gram_matvec(kernel_)

        def cov_matvec(v):
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        # Evaluate the log-pdf
        precon, info_p = precondition(kernel_)
        value, info_l = logpdf_p(
            targets, *p_logpdf, mean=mean_, cov_matvec=cov_matvec, P=precon
        )

        # Normalise by the number of data points because GPyTorch does
        return value / len(inputs), {"precondition": info_p, "logpdf": info_l}

    return mll


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
            # not differentiable at zero)
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


def condition(prior, likelihood, *, solve: Callable, gram_matvec: Callable):
    def posterior(xs, *, inputs, targets, params_prior: dict, params_likelihood: dict):
        # Compute (K + s^2 I)^{-1} y
        k_inv_times_y, _info = _representer_weights(
            inputs,
            targets,
            params_prior=params_prior,
            params_likelihood=params_likelihood,
        )

        # Evaluate the mean at the new gridpoint
        mean_posterior = _mean(
            xs, weights=k_inv_times_y, inputs=inputs, params_prior=params_prior
        )

        # Eventually, we might replace the "None" with a covariance...
        return mean_posterior, None

    def _representer_weights(inputs, targets, params_prior, params_likelihood):
        mean, kernel_prior = prior(inputs, params=params_prior)
        mean_, kernel_likelihood = likelihood(
            mean, kernel_prior, params=params_likelihood
        )

        # Build matvec for likelihood

        def cov_matvec_likelihood(v):
            cov = gram_matvec(kernel_likelihood)

            i = jnp.arange(len(inputs))
            return cov(i, i, v)

        return solve(cov_matvec_likelihood, targets - mean_)

    def _mean(xs, /, weights, *, inputs, params_prior):
        prior_mean, kernel_prior = prior(xs, params=params_prior)

        def cov_matvec_prior(v):
            cov = gram_matvec(kernel_prior)
            i = jnp.arange(len(xs))
            j = jnp.arange(len(inputs))
            return cov(i, j, v)

        return prior_mean + cov_matvec_prior(weights)

    return posterior


def _assert_shapes(x, y, shape_in):
    if jnp.shape(x) != jnp.shape(y):
        error = "The arguments have different shapes: "
        error += f"{jnp.shape(x)} != {jnp.shape(y)})"
        raise ValueError(error)

    if jnp.shape(x) != shape_in:
        error = f"The shape {jnp.shape(x)} of the first argument "
        error += "does not match 'shape_in'={shape_in}"
        raise ValueError(error)
