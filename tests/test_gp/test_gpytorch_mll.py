"""Tests for marginal-log-likelihood models."""

import gpytorch
import jax
import jax.numpy as jnp
import pytest_cases
import torch
from matfree_extensions import gp


@pytest_cases.case
def case_logpdf_scipy_stats():
    return gp.logpdf_scipy_stats(), ()


@pytest_cases.case
def case_logpdf_cholesky():
    return gp.logpdf_cholesky(), ()


@pytest_cases.case
@pytest_cases.parametrize("samples", ["rademacher", "normal"])
def case_logpdf_lanczos(samples):
    num = 100_000  # maaaany samples because we test for exactness
    krylov_depth = 2  # because the number of data points is 3

    logpdf = gp.logpdf_lanczos(
        krylov_depth, slq_sample_num=num, slq_sample_type=samples
    )
    params = (jax.random.PRNGKey(1),)
    return logpdf, params


@pytest_cases.parametrize_with_cases("logpdf", cases=".")
def test_mll_exact(logpdf):
    # Compute the reference model
    reference = _model_and_mll_via_gpytorch()
    (x, y), value_ref, ((lengthscale, outputscale), noise) = reference

    # Log-pdf function
    logpdf_fun, p_logpdf = logpdf

    # Set up a GP model
    k, p_prior = gp.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = gp.model(gp.mean_zero(), k)
    likelihood, p_likelihood = gp.likelihood_gaussian()
    loss = gp.mll_exact(prior, likelihood, logpdf=logpdf_fun)

    # Ensure that the parameters match
    p_prior["raw_lengthscale"] = lengthscale.squeeze()
    p_prior["raw_outputscale"] = outputscale.squeeze()
    p_likelihood["raw_noise"] = noise.squeeze()

    # Evaluate the MLL
    value = loss(x, y, *p_logpdf, params_prior=p_prior, params_likelihood=p_likelihood)

    # Assert that the values match
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(value)).eps)
    assert jnp.allclose(value_ref, value, rtol=small_value, atol=small_value)


class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_):
        super().__init__(train_x, train_y, likelihood_)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _model_and_mll_via_gpytorch():
    """Compute the MLL from the GPyTorch reference."""
    x = torch.arange(1.0, 4.0)
    y = x

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = _ExactGPModel(x, y, likelihood)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    value = mll(model(x), y)

    lengthscale = jnp.asarray(
        model.covar_module.base_kernel.raw_lengthscale.detach().numpy()
    )
    outputscale = jnp.asarray(model.covar_module.raw_outputscale.detach().numpy())
    noise = jnp.asarray(likelihood.raw_noise.detach().numpy())
    params = ((lengthscale, outputscale), noise)

    x = jnp.asarray(x.detach().numpy())
    y = jnp.asarray(y.detach().numpy())
    value = jnp.asarray(value.detach().numpy())
    return (x, y), value, params
