"""Gaussian process models."""

import functools
from typing import Callable

import jax
import jax.numpy as jnp
from matfree import hutchinson

from matfree_extensions import lanczos

# todo: implement a logpdf with a custom vjp that reuses a CG call?!


def target_logml(model: Callable, likelihood: Callable, /) -> Callable:
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
        loss = likelihood(inputs, mean=mean, kernel=kernel, params=params_likelihood)
        value, info_pdf = loss(targets, *p_logpdf)
        return value, info_pdf

    return mll


def target_posterior(model: Callable, likelihood: Callable, /) -> Callable:
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
    constrain = constraint_greater_than(0.0)

    def parametrize(*, raw_lengthscale, raw_outputscale):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            # Apply a soft-plus because GPyTorch does
            lengthscale = constrain(raw_lengthscale)
            outputscale = constrain(raw_outputscale)

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
    constrain = constraint_greater_than(0.0)

    def parametrize(*, raw_lengthscale, raw_outputscale):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            # Apply a soft-plus because GPyTorch does
            lengthscale = constrain(raw_lengthscale)
            outputscale = constrain(raw_outputscale)

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
    constrain = constraint_greater_than(0.0)

    def parametrize(*, raw_lengthscale, raw_outputscale):
        def k(x, y):
            _assert_shapes(x, y, shape_in)

            # Apply a soft-plus because GPyTorch does
            lengthscale = constrain(raw_lengthscale)
            outputscale = constrain(raw_outputscale)

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


def constraint_greater_than(minval, /):
    def softplus(x, beta=1.0, threshold=20.0):
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

    return lambda s: minval + softplus(s)


def _assert_shapes(x, y, shape_in):
    if jnp.shape(x) != jnp.shape(y):
        error = "The arguments have different shapes: "
        error += f"{jnp.shape(x)} != {jnp.shape(y)})"
        raise ValueError(error)

    if jnp.shape(x) != shape_in:
        error = f"The shape {jnp.shape(x)} of the first argument "
        error += "does not match 'shape_in'={shape_in}"
        raise ValueError(error)


def likelihood_pdf(
    matvec: Callable, logpdf: Callable, *, constrain: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(inputs, mean: Callable, kernel: Callable, params: dict):
        raw_noise = params["raw_noise"]
        noise = constrain(raw_noise)

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        def logpdf_partial(targets, *p_logpdf):
            mean_array = jax.vmap(mean)(inputs)
            return logpdf(targets, *p_logpdf, mean=mean_array, cov_matvec=cov_matvec)

        return logpdf_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def likelihood_pdf_p(
    matvec: Callable, logpdf_p: Callable, precondition: Callable, *, constrain: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(inputs, mean: Callable, kernel: Callable, params: dict):
        raw_noise = params["raw_noise"]
        noise = constrain(raw_noise)

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j])

        def cov_matvec(v):
            cov = matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        pre, info = precondition(lazy_kernel, len(inputs))

        def logpdf_partial(targets, *p_logpdf):
            mean_array = jax.vmap(mean)(inputs)
            val, aux = logpdf_p(
                targets,
                *p_logpdf,
                mean=mean_array,
                cov_matvec=lambda v: cov_matvec(v) + noise * v,
                P=lambda v: pre(v, noise),
            )
            return val, {"precondition": info, "logpdf": aux}

        return logpdf_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def likelihood_condition(
    matvec: Callable, solve: Callable, *, constrain: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(inputs, mean: Callable, kernel: Callable, params: dict):
        raw_noise = params["raw_noise"]
        noise = constrain(raw_noise)

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j]) + noise * (i == j)

        def cov_matvec(v):
            cov = matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        def condition_partial(xs, targets):
            mean_array = jax.vmap(mean)(inputs)
            weights, info = solve(cov_matvec, targets - mean_array)

            def cov_matvec_prior(v):
                cov = matvec(kernel)
                return cov(xs, inputs, v)

            mean_eval = jax.vmap(mean)(xs)
            return mean_eval + cov_matvec_prior(weights), {"solve": info}

        return condition_partial

    p = {"raw_noise": jnp.empty(())}
    return likelihood, p


def likelihood_condition_p(
    matvec: Callable, solve_p: Callable, *, precondition: Callable, constrain: Callable
) -> tuple[Callable, dict]:
    """Construct a Gaussian likelihood."""

    def likelihood(inputs: jax.Array, mean: Callable, kernel: Callable, params: dict):
        raw_noise = params["raw_noise"]
        noise = constrain(raw_noise)

        def lazy_kernel(i: int, j: int):
            return kernel(inputs[i], inputs[j])

        def cov_matvec(v):
            cov = matvec(lazy_kernel)
            idx = jnp.arange(len(inputs))
            return cov(idx, idx, v)

        pre, info = precondition(lazy_kernel, len(inputs))

        def condition_partial(xs, targets):
            mean_array = jax.vmap(mean)(inputs)

            weights, info = solve_p(
                lambda v: cov_matvec(v) + noise * v,
                targets - mean_array,
                P=lambda v: pre(v, noise),
            )

            def cov_matvec_prior(v):
                cov = matvec(kernel)
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


def gram_matvec_sequential(*, checkpoint: bool):
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


def gram_matvec_partitioned(num: int, *, checkpoint: bool):
    """Turn a covariance function into a gram-matrix vector product.

    Compute the matrix-vector product by mapping over full batches.
    Why? To reduce memory compared to gram_matvec,
    but to increase runtime compare to gram_matvec_sequential.

    Parameters
    ----------
    num
        Number of partitions. Make this value as large as possible,
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
            ndata, *shape = jnp.shape(i)
            if ndata % num != 0:
                msg = f"num = {num} does not divide dataset size {ndata}."
                raise ValueError(msg)

            mv = matvec_single(j, v)
            if checkpoint:
                mv = jax.checkpoint(mv)

            x_batched = jnp.reshape(i, (num, ndata // num, *shape))
            mapped = jax.lax.map(mv, x_batched)
            return jnp.reshape(mapped, (-1,))

        matvec_dense = gram_matvec()
        matvec_dense_f = matvec_dense(fun)

        def matvec_single(j, v):
            def mv(x_batched):
                return matvec_dense_f(x_batched, j, v)

            return mv

        return matvec_map

    return matvec


def gram_matvec():
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


def krylov_logdet_slq(
    krylov_depth, /, *, sample: Callable, num_batches: int, checkpoint: bool
):
    def logdet(A: Callable, /, key):
        integrand = lanczos.integrand_spd(jnp.log, krylov_depth, A)
        estimate = hutchinson.hutchinson(integrand, sample)

        # If a single batch, we never checkpoint.
        if num_batches == 1:
            value = estimate(key)
            return value, {"std": 0.0, "std_rel": 0.0}

        # Memory-efficient reverse-mode derivatives
        #  See gram_matvec_map_over_batch().

        if checkpoint:
            estimate = jax.checkpoint(estimate)

        keys = jax.random.split(key, num=num_batches)
        values = jax.lax.map(lambda k: estimate(k), keys)
        mean = jnp.mean(values, axis=0)
        std = jnp.std(values, axis=0)
        return mean, {"std_abs": std, "std_rel": std / jnp.abs(mean)}

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
        mean = jnp.mean(values, axis=0)
        std = jnp.std(values, axis=0)
        return mean, {"std": std}

    return logdet
