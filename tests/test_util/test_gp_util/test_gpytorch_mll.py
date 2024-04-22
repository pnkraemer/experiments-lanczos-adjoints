"""Tests for marginal-log-likelihood models."""

import gpytorch
import jax
import jax.numpy as jnp
import pytest_cases
import torch
from matfree import hutchinson
from matfree_extensions.util import gp_util


@pytest_cases.case
def case_logpdf_scipy_stats():
    return gp_util.logpdf_scipy_stats(), ()


@pytest_cases.case
def case_logpdf_cholesky():
    return gp_util.logpdf_cholesky(), ()


@pytest_cases.case
def case_logpdf_lanczos_rademacher():
    # maaaany samples because we test for exactness
    num_batches, num_samples = 2, 50_000

    # Max order (the number of data points is 3)
    krylov_depth = 2

    x_like = jnp.ones((3,), dtype=float)
    sampler = hutchinson.sampler_rademacher(x_like, num=num_samples)
    logpdf = gp_util.logpdf_lanczos(krylov_depth, sampler, slq_batch_num=num_batches)
    params = (jax.random.PRNGKey(1),)
    return logpdf, params


@pytest_cases.case
def case_logpdf_lanczos_normal():
    # maaaany samples because we test for exactness
    num_batches, num_samples = 2, 50_000

    # Max order (the number of data points is 3)
    krylov_depth = 2

    x_like = jnp.ones((3,), dtype=float)
    sampler = hutchinson.sampler_normal(x_like, num=num_samples)
    logpdf = gp_util.logpdf_lanczos(krylov_depth, sampler, slq_batch_num=num_batches)
    params = (jax.random.PRNGKey(1),)
    return logpdf, params


@pytest_cases.case
def case_logpdf_lanczos_rademacher_reuse():
    # maaaany samples because we test for exactness
    num_batches, num_samples = 2, 50_000

    # Max order (the number of data points is 3)
    krylov_depth = 2

    x_like = jnp.ones((3,), dtype=float)
    sampler = hutchinson.sampler_rademacher(x_like, num=num_samples)
    logpdf = gp_util.logpdf_lanczos_reuse(
        krylov_depth, sampler, slq_batch_num=num_batches
    )
    params = (jax.random.PRNGKey(1),)
    return logpdf, params


@pytest_cases.case
def case_gram_matvec_full_batch():
    return gp_util.gram_matvec_full_batch()


@pytest_cases.case
def case_gram_matvec_map_over_batch():
    return gp_util.gram_matvec_map_over_batch(batch_size=1)


@pytest_cases.case
def case_gram_matvec_map():
    return gp_util.gram_matvec_map()


@pytest_cases.parametrize_with_cases("logpdf", cases=".", prefix="case_logpdf_")
@pytest_cases.parametrize_with_cases("gram_matvec", cases=".", prefix="case_gram_")
def test_mll_exact(logpdf, gram_matvec):
    # Compute the reference model
    reference = _model_and_mll_via_gpytorch()
    (x, y), value_ref, ((lengthscale, outputscale), noise) = reference

    # Log-pdf function
    logpdf_fun, p_logpdf = logpdf

    # Set up a GP model
    k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = gp_util.model(gp_util.mean_zero(), k, gram_matvec=gram_matvec)
    likelihood, p_likelihood = gp_util.likelihood_gaussian()
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf_fun)

    # Ensure that the parameters match
    p_prior["raw_lengthscale"] = lengthscale.squeeze()
    p_prior["raw_outputscale"] = outputscale.squeeze()
    p_likelihood["raw_noise"] = noise.squeeze()

    # Evaluate the MLL
    # We do it in a weird way, by computing value_and_grad.
    # The reason is that we need to ensure that the MLL function
    # is differentiable, and this is the most obvious place for doing so.
    def mll(p1, p2):
        return loss(x, y, *p_logpdf, params_prior=p1, params_likelihood=p2)

    value, _grad = jax.value_and_grad(mll)(p_prior, p_likelihood)

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