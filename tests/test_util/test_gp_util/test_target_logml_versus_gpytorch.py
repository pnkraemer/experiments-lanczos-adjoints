"""Tests for marginal-log-likelihood models."""

import gpytorch
import jax
import jax.numpy as jnp
import pytest_cases
import torch
from matfree import hutchinson
from matfree_extensions import cg, low_rank
from matfree_extensions.util import gp_util


@pytest_cases.case
def case_logpdf_scipy_stats():
    return gp_util.logpdf_scipy_stats(), ()


@pytest_cases.case
def case_logpdf_cholesky():
    return gp_util.logpdf_cholesky(), ()


@pytest_cases.case
def case_logpdf_krylov():
    # maaaany samples because we test for exactness
    num_batches, num_samples = 2, 50_000

    # Max order (the number of data points is 3)
    krylov_depth = 2

    x_like = jnp.ones((3,), dtype=float)
    sample = hutchinson.sampler_rademacher(x_like, num=num_samples)

    solve = cg.cg_adaptive(atol=1e-4, rtol=0.0, maxiter=1000)
    logdet = gp_util.krylov_logdet_slq(
        krylov_depth, sample=sample, num_batches=num_batches, checkpoint=False
    )
    logpdf = gp_util.logpdf_krylov(solve=solve, logdet=logdet)
    params = (jax.random.PRNGKey(1),)
    return logpdf, params


@pytest_cases.case
def case_logpdf_krylov_reuse():
    # maaaany samples because we test for exactness
    num_batches, num_samples = 2, 50_000

    # Max order (the number of data points is 3)
    krylov_depth = 2

    x_like = jnp.ones((3,), dtype=float)
    sample = hutchinson.sampler_rademacher(x_like, num=num_samples)
    solve = cg.cg_adaptive(atol=1e-4, rtol=0.0, maxiter=1000)
    logdet = gp_util.krylov_logdet_slq_vjp_reuse(
        krylov_depth, sample=sample, num_batches=num_batches, checkpoint=False
    )
    logpdf = gp_util.logpdf_krylov(solve, logdet)
    params = (jax.random.PRNGKey(1),)
    return logpdf, params


@pytest_cases.case
def case_precon_logpdf_krylov():
    # maaaany samples because we test for exactness
    num_batches, num_samples = 2, 50_000

    # Max order (the number of data points is 3)
    krylov_depth = 2

    x_like = jnp.ones((3,), dtype=float)
    sample = hutchinson.sampler_rademacher(x_like, num=num_samples)
    logdet = gp_util.krylov_logdet_slq(
        krylov_depth, sample=sample, num_batches=num_batches, checkpoint=False
    )

    solve = cg.pcg_fixed_step(3)
    logpdf = gp_util.logpdf_krylov_p(solve_p=solve, logdet=logdet)
    params = (jax.random.PRNGKey(1),)
    return logpdf, params


@pytest_cases.case
def case_gram_matvec_full_batch():
    return gp_util.gram_matvec()


@pytest_cases.case
def case_gram_matvec_partitioned_checkpoint():
    return gp_util.gram_matvec_partitioned(1, checkpoint=True)


@pytest_cases.case
def case_gram_matvec_sequential():
    return gp_util.gram_matvec_sequential(checkpoint=False)


@pytest_cases.case
def case_preconditioner_partial_cholesky():
    cholesky = low_rank.cholesky_partial(rank=1)
    return low_rank.preconditioner(cholesky)


@pytest_cases.case
def case_preconditioner_partial_cholesky_pivot():
    cholesky = low_rank.cholesky_partial_pivot(rank=1)
    return low_rank.preconditioner(cholesky)


@pytest_cases.parametrize_with_cases("logpdf", cases=".", prefix="case_logpdf_")
@pytest_cases.parametrize_with_cases("gram_matvec", cases=".", prefix="case_gram_")
def test_logml_matches_gpytorch_mll(logpdf, gram_matvec):
    # Compute the reference model
    reference = _model_and_mll_via_gpytorch()
    (x, y), value_ref, ((lengthscale, outputscale), noise) = reference

    # Log-pdf function
    logpdf_fun, p_logpdf = logpdf

    # Set up a GP model
    m, p_mean = gp_util.mean_constant(shape_out=())
    k, p_kernel = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = gp_util.model_gp(m, k)
    constrain = gp_util.constraint_greater_than(1e-4)
    likelihood, p_likelihood = gp_util.likelihood_pdf(
        gram_matvec, logpdf_fun, constrain=constrain
    )
    loss = gp_util.target_logml(prior, likelihood)

    # Ensure that the parameters match
    p_mean["constant_value"] = 0.0
    p_kernel["raw_lengthscale"] = lengthscale.squeeze()
    p_kernel["raw_outputscale"] = outputscale.squeeze()
    p_likelihood["raw_noise"] = noise.squeeze()

    # Evaluate the MLL
    # We do it in a weird way, by computing value_and_grad.
    # The reason is that we need to ensure that the MLL function
    # is differentiable, and this is the most obvious place for doing so.
    def mll(p1, p2, p3):
        return loss(
            x, y, *p_logpdf, params_mean=p1, params_kernel=p2, params_likelihood=p3
        )

    (value, _info), _grad = jax.value_and_grad(mll, has_aux=True)(
        p_mean, p_kernel, p_likelihood
    )

    # Assert that the values match
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(value)).eps)
    assert jnp.allclose(value_ref, value / len(x), rtol=small_value, atol=small_value)


@pytest_cases.parametrize_with_cases(
    "logpdf_p", cases=".", prefix="case_precon_logpdf_"
)
@pytest_cases.parametrize_with_cases("gram_matvec", cases=".", prefix="case_gram_")
@pytest_cases.parametrize_with_cases("precon", cases=".", prefix="case_preconditioner_")
def test_logml_matches_gpytorch_mll_preconditioned(logpdf_p, gram_matvec, precon):
    # Compute the reference model
    reference = _model_and_mll_via_gpytorch()
    (x, y), value_ref, ((lengthscale, outputscale), noise) = reference

    # Log-pdf function
    logpdf_fun, p_logpdf = logpdf_p

    # Set up a GP model
    m, p_mean = gp_util.mean_constant(shape_out=())
    k, p_kernel = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = gp_util.model_gp(m, k)

    constrain = gp_util.constraint_greater_than(1e-4)
    likelihood, p_likelihood = gp_util.likelihood_pdf_p(
        gram_matvec, logpdf_fun, precondition=precon, constrain=constrain
    )
    loss = gp_util.target_logml(prior, likelihood)

    # Ensure that the parameters match
    p_mean["constant_value"] = 0.0
    p_kernel["raw_lengthscale"] = lengthscale.squeeze()
    p_kernel["raw_outputscale"] = outputscale.squeeze()
    p_likelihood["raw_noise"] = noise.squeeze()

    # Evaluate the MLL
    # We do it in a weird way, by computing value_and_grad.
    # The reason is that we need to ensure that the MLL function
    # is differentiable, and this is the most obvious place for doing so.
    def mll(p1, p2, p3):
        return loss(
            x, y, *p_logpdf, params_mean=p1, params_kernel=p2, params_likelihood=p3
        )

    (value, _info), _grad = jax.value_and_grad(mll, has_aux=True)(
        p_mean, p_kernel, p_likelihood
    )

    # Assert that the values match
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(value)).eps)
    assert jnp.allclose(value_ref, value / len(x), rtol=small_value, atol=small_value)


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
