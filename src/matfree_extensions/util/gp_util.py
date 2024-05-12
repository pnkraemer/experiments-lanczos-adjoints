"""Gaussian process models."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp

# todo: implement a logpdf with a custom vjp that reuses a CG call?!


def target_logml(model: Callable, /, likelihood_pdf: Callable) -> Callable:
    """Construct a log-marginal-likelihood function."""

    def mll(
        inputs,
        targets,
        *p_logpdf,
        params_mean: dict,
        params_kernel: dict,
        params_likelihood: dict,
    ):
        # Evaluate the marginal data likelihood
        mean, kernel = model(params_mean=params_mean, params_kernel=params_kernel)
        loss = likelihood_pdf(
            inputs, mean=mean, kernel=kernel, params=params_likelihood
        )
        value, info_pdf = loss(targets, *p_logpdf)
        return value, info_pdf

    return mll


def target_posterior(model: Callable, /, likelihood: Callable) -> Callable:
    """Construct a marginal log-likelihood function."""

    def posterior(
        inputs, targets, params_mean: dict, params_kernel: dict, params_likelihood: dict
    ):
        mean, kernel = model(params_mean, params_kernel)
        condition = likelihood(inputs, mean, kernel, params=params_likelihood)
        return functools.partial(condition, targets=targets), {}

    return posterior


def model_gp(mean_fun: Callable, kernel_fun: Callable) -> Callable:
    """Construct a Gaussian process model."""

    def prior(params_mean: dict, params_kernel: dict):
        mean = mean_fun(**params_mean)
        kernel = kernel_fun(**params_kernel)
        return mean, kernel

    return prior


def mean_constant(*, shape_out) -> tuple[Callable, dict]:
    """Construct a zero mean-function."""

    def parametrize(*, constant_value):
        return lambda _x: constant_value

    p_mean = {"constant_value": jnp.empty(shape_out)}
    return parametrize, p_mean


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


def likelihood_pdf(gram_matvec: Callable, logpdf: Callable) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(inputs, mean: Callable, kernel: Callable, params: dict):
        # Apply a soft-plus because GPyTorch does
        # todo: implement a box-constraint?
        raw_noise = params["raw_noise"]
        noise = _softplus(raw_noise)

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = gram_matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        def logpdf_partial(targets, *p_logpdf):
            mean_array = jax.vmap(mean)(inputs)
            return logpdf(targets, *p_logpdf, mean=mean_array, cov_matvec=cov_matvec)

        return logpdf_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def likelihood_pdf_p(
    gram_matvec: Callable, logpdf_p: Callable, precondition: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(inputs, mean: Callable, kernel: Callable, params: dict):
        # Apply a soft-plus because GPyTorch does
        raw_noise = params["raw_noise"]
        noise = _softplus(raw_noise)

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = gram_matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        pre, info = precondition(lazy_kernel, len(inputs))

        def logpdf_partial(targets, *p_logpdf):
            mean_array = jax.vmap(mean)(inputs)
            val, aux = logpdf_p(
                targets,
                *p_logpdf,
                mean=mean_array,
                cov_matvec=cov_matvec,
                P=lambda v: pre(v, noise),
            )
            return val, {"precondition": info, "logpdf": aux}

        return logpdf_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def likelihood_condition(
    gram_matvec: Callable, solve: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(inputs, mean: Callable, kernel: Callable, params: dict):
        # Apply a soft-plus because GPyTorch does
        raw_noise = params["raw_noise"]
        noise = _softplus(raw_noise)

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = gram_matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        def condition_partial(xs, targets):
            mean_array = jax.vmap(mean)(inputs)
            weights, info = solve(cov_matvec, targets - mean_array)

            def cov_matvec_prior(v):
                cov = gram_matvec(kernel)
                return cov(xs, inputs, v)

            mean_eval = jax.vmap(mean)(xs)
            return mean_eval + cov_matvec_prior(weights), {"solve": info}

        return condition_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


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
        logdet_, info_logdet = logdet(cov_matvec, *params_logdet)
        logdet_ /= 2

        # Mahalanobis norm
        tmp, info_solve = solve(cov_matvec, y - mean)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        info = {"logdet": info_logdet, "solve": info_solve}
        (n,) = jnp.shape(mean)
        return -logdet_ - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), info

    return logpdf


def logpdf_krylov_p(solve_p: Callable, logdet: Callable):
    """Evaluate a logpdf via preconditioned Krylov methods."""

    def logpdf(y, *params_logdet, mean, cov_matvec, P: Callable):
        # Log-determinant
        logdet_, info_logdet = logdet(cov_matvec, *params_logdet)
        logdet_ /= 2

        # Mahalanobis norm
        tmp, info_solve = solve_p(cov_matvec, y - mean, P=P)
        mahalanobis = jnp.dot(y - mean, tmp)

        # Combine the terms
        info = {"logdet": info_logdet, "solve": info_solve}
        (n,) = jnp.shape(mean)
        return -logdet_ - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), info

    return logpdf
