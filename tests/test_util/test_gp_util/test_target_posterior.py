"""Tests for log-marginal likelihoods."""

import jax
import jax.numpy as jnp
from matfree_extensions.util import gp_util


def test_posterior_mean_interpolates_data_if_noise_is_small(n=100):
    # Set up: data
    xs = jnp.linspace(0, 1, num=n)
    ys = jnp.linspace(0, 1, num=n)

    # Set up linear algebra

    def solve(A, b):
        matrix = jax.jacfwd(A)(b)
        return jnp.linalg.solve(matrix, b), {}

    # Set up: model
    m, p_mean = gp_util.mean_constant(shape_out=())
    k, p_kernel = gp_util.kernel_scaled_matern_32(shape_in=(), shape_out=())
    prior = gp_util.model_gp(m, k)
    gram_matvec = gp_util.gram_matvec()
    constrain = gp_util.constraint_greater_than(1e-4)
    likelihood, p_likelihood = gp_util.likelihood_condition(
        gram_matvec, solve=solve, constrain=constrain
    )
    p_likelihood["raw_noise"] = -10.0

    # Evaluate the posterior
    posterior = gp_util.target_posterior(prior, likelihood)
    mean, _ = posterior(
        xs,
        ys,
        params_mean=p_mean,
        params_kernel=p_kernel,
        params_likelihood=p_likelihood,
    )
    tol = jnp.sqrt(_softplus(p_likelihood["raw_noise"]))
    posterior_mean, _ = mean(xs[: len(ys) // 2])
    assert jnp.allclose(posterior_mean, ys[: len(ys) // 2], atol=tol, rtol=tol)


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
