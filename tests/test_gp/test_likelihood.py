"""Tests for likelihood models."""

import jax
import jax.numpy as jnp
from matfree_extensions import gp


def mean_zero():
    def mean(x):
        return jnp.zeros((len(x),), dtype=jnp.dtype(x))

    return mean


def model_gp(mean_fun, kernel_fun):
    def prior(x, **kernel_params):
        mean = mean_fun(x)
        cov = kernel_fun(**kernel_params)(x, x)
        return mean, cov

    return prior


def likelihood_gaussian():
    def likelihood(mean, cov, *, standard_deviation):
        # todo: softmax the standard deviation?
        return mean, cov + jnp.eye(len(cov)) * standard_deviation

    p = {"standard_deviation": jnp.empty(())}
    return likelihood, p


def mll_exact_gp(prior, likelihood):
    def mll(x, y, params_prior: dict, params_likelihood: dict):
        mean, cov = prior(x, **params_prior)
        mean_, cov_ = likelihood(mean, cov, **params_likelihood)
        return jax.scipy.stats.multivariate_normal.logpdf(y, mean=mean_, cov=cov_)

    return mll


def test_mll_exact():
    # Set up a GP model
    k, p_prior = gp.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = model_gp(mean_zero(), k)
    likelihood, p_likelihood = likelihood_gaussian()
    loss = mll_exact_gp(prior, likelihood)

    # Evaluate the MLL
    x = jnp.arange(1.0, 4.0)
    y = x
    value = loss(x, y, params_prior=p_prior, params_likelihood=p_likelihood)
    print(value)

    assert False
