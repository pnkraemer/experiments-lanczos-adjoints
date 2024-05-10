"""Tests for log-marginal likelihoods."""

import jax
import jax.numpy as jnp
from matfree_extensions.util import gp_util, gp_util_linalg


def test_predict_posterior(n=100):
    # Set up: data
    xs = jnp.linspace(0, 1, num=n)
    ys = jnp.linspace(0, 1, num=n)

    # Set up: model
    k, p_prior = gp_util.kernel_scaled_matern_32(shape_in=(), shape_out=())
    prior = gp_util.model(gp_util.mean_zero(), k)
    likelihood, p_likelihood = gp_util.likelihood_gaussian()
    p_likelihood["raw_noise"] = -10.0

    # Set up linear algebra
    gram_matvec = gp_util_linalg.gram_matvec_full_batch()

    def solve(A, b):
        matrix = jax.jacfwd(A)(b)
        return jnp.linalg.solve(matrix, b), {}

    # Predict
    posterior = gp_util.condition(
        prior, likelihood, solve=solve, gram_matvec=gram_matvec
    )
    mean, _lazy_kernel = posterior(
        xs, inputs=xs, targets=ys, params_prior=p_prior, params_likelihood=p_likelihood
    )

    tol = jnp.sqrt(_softplus(p_likelihood["raw_noise"]))
    assert jnp.allclose(mean, ys, atol=tol, rtol=tol)


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
