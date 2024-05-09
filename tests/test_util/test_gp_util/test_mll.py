"""Tests for log-marginal likelihoods."""

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import hutchinson
from matfree_extensions.util import gp_util


@pytest_cases.parametrize(
    "low_rank", [gp_util.low_rank_cholesky_pivot, gp_util.low_rank_cholesky]
)
def test_preconditioning_reduces_cg_iteration_count(
    low_rank, n=100, rank=2, tol=1e-4, max_steps=100
):
    # Set up a testproblem
    xs = jnp.linspace(0, 1, num=n)
    ys = jnp.linspace(0, 1, num=n)
    k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
    likelihood, p_likelihood = gp_util.likelihood_gaussian()

    # Set up all constant parameters
    gram_matvec = gp_util.gram_matvec_full_batch()
    v_like = xs
    sample = hutchinson.sampler_normal(v_like, num=10)
    logdet = gp_util.krylov_logdet_slq(5, sample=sample, num_batches=1)

    # [NO PRECON] Set up a solver
    solve = gp_util.krylov_solve_cg_lineax(atol=tol, rtol=tol, max_steps=max_steps)
    logpdf_fun = gp_util.logpdf_krylov(solve=solve, logdet=logdet)

    # [NO PRECON] Set up a GP model
    prior = gp_util.model(gp_util.mean_zero(), k, gram_matvec=gram_matvec)
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf_fun)

    # [NO PRECON] Evaluate the log-likelihood
    key = jax.random.PRNGKey(1)
    value, info = loss(
        xs, ys, key, params_prior=p_prior, params_likelihood=p_likelihood
    )
    num_steps_without = info["num_steps"]
    value_without = value

    # [PRECON] Set up a solver
    solve = gp_util.krylov_solve_cg_lineax_precondition(
        atol=tol, rtol=tol, max_steps=max_steps
    )
    logpdf_fun = gp_util.logpdf_krylov(solve=solve, logdet=logdet)

    # [PRECON] Set up a GP model
    low_rank_impl = low_rank(n, rank=rank)
    P = gp_util.precondition_low_rank(low_rank_impl, small_value=1e-4)
    prior = gp_util.model_precondition(
        gp_util.mean_zero(), k, gram_matvec=gram_matvec, precondition=P
    )
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf_fun)

    # [NO PRECON] Evaluate the log-likelihood
    key = jax.random.PRNGKey(1)
    value, info = loss(
        xs, ys, key, params_prior=p_prior, params_likelihood=p_likelihood
    )
    num_steps_with = info["num_steps"]
    value_with = value

    # Compare values
    assert num_steps_with < num_steps_without
    assert jnp.allclose(value_with, value_without, atol=1e-1, rtol=1e-1)
