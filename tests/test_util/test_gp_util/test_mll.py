"""Tests for log-marginal likelihoods."""

import jax
import jax.numpy as jnp
from matfree import hutchinson
from matfree_extensions.util import gp_util


def test_preconditioning():
    xs = jnp.linspace(0, 1)
    ys = jnp.linspace(0, 1)

    gram_matvec = gp_util.gram_matvec_full_batch()

    v_like = xs
    sample = hutchinson.sampler_normal(v_like, num=10)
    logdet = gp_util.krylov_logdet_slq(5, sample=sample, num_batches=1)
    solve = gp_util.krylov_solve_cg_lineax(atol=0.1, rtol=0.1, max_steps=100)
    logpdf_fun = gp_util.logpdf_krylov(solve=solve, logdet=logdet)

    # Set up a GP model
    k, p_prior = gp_util.kernel_scaled_matern_32(shape_in=(), shape_out=())
    prior = gp_util.model(gp_util.mean_zero(), k, gram_matvec=gram_matvec)
    likelihood, p_likelihood = gp_util.likelihood_gaussian()
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf_fun)

    # Evaluate the log-likelihood
    key = jax.random.PRNGKey(1)
    value, info = loss(
        xs, ys, key, params_prior=p_prior, params_likelihood=p_likelihood
    )
    print(value)
    print(info)
    assert False
