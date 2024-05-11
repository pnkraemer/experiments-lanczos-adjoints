"""Tests for log-marginal likelihoods."""

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import hutchinson
from matfree_extensions import low_rank
from matfree_extensions.util import gp_util, gp_util_linalg


@pytest_cases.parametrize(
    "cholesky", [low_rank.cholesky_partial_pivot, low_rank.cholesky_partial]
)
def test_preconditioning_reduces_cg_error(cholesky, n=100, rank=10, num_steps=10):
    # Set up: data
    xs = jnp.linspace(0, 1, num=n)
    ys = jnp.linspace(0, 1, num=n)

    # Set up: model
    k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = gp_util.model(gp_util.mean_zero(), k)
    likelihood, p_likelihood = gp_util.likelihood_gaussian()

    # Set up: linear algebra
    v_like = xs
    sample = hutchinson.sampler_normal(v_like, num=10)
    logdet = gp_util_linalg.krylov_logdet_slq(5, sample=sample, num_batches=1)
    gram_matvec = gp_util_linalg.gram_matvec_full_batch()

    # No precon:
    solve = gp_util_linalg.krylov_solve_cg_fixed_step(num_steps)
    logpdf = gp_util.logpdf_krylov(solve=solve, logdet=logdet)
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf, gram_matvec=gram_matvec)

    # Precon:
    solve_p = gp_util_linalg.krylov_solve_pcg_fixed_step(num_steps)
    logpdf_p = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)
    cholesky_impl = cholesky(n, rank=rank)
    precon = low_rank.preconditioner(cholesky_impl, small_value=1e-10)
    loss_p = gp_util.mll_exact_p(
        prior,
        likelihood,
        logpdf_p=logpdf_p,
        gram_matvec=gram_matvec,
        precondition=precon,
    )

    # Call both value functions
    key = jax.random.PRNGKey(1)
    value, info = loss(
        xs, ys, key, params_prior=p_prior, params_likelihood=p_likelihood
    )
    value_p, info_p = loss_p(
        xs, ys, key, params_prior=p_prior, params_likelihood=p_likelihood
    )

    def res(x):
        return jnp.linalg.norm(x) / jnp.sqrt(x.size)

    # Compare values
    print(info_p["logpdf"]["residual"])
    print(info["logpdf"]["residual"])
    print(value_p, value)
    assert jnp.allclose(value_p, value, atol=1e-1, rtol=1e-1)

    assert res(info_p["logpdf"]["residual"]) < res(info["logpdf"]["residual"])
    assert False


@pytest_cases.parametrize(
    "cholesky", [low_rank.cholesky_partial_pivot, low_rank.cholesky_partial]
)
def test_mll_remains_differentiable_despite_preconditioning(
    cholesky, n=100, rank=2, tol=1e-4, maxiter=100
):
    # Set up a testproblem
    xs = jnp.linspace(0, 1, num=n)
    ys = jnp.linspace(0, 1, num=n)

    # Set up a model
    k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = gp_util.model(gp_util.mean_zero(), k)
    likelihood, p_likelihood = gp_util.likelihood_gaussian()

    # Set up linear algebra
    v_like = xs
    sample = hutchinson.sampler_normal(v_like, num=10)
    logdet = gp_util_linalg.krylov_logdet_slq(5, sample=sample, num_batches=1)
    solve_p = gp_util_linalg.krylov_solve_pcg_jax(tol=tol, maxiter=maxiter)
    gram_matvec = gp_util_linalg.gram_matvec_full_batch()
    logpdf_p = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)

    # Set up an MLL
    cholesky_impl = cholesky(n, rank=rank)
    precondition = low_rank.preconditioner(cholesky_impl, small_value=1e-4)
    loss = gp_util.mll_exact_p(
        prior,
        likelihood,
        logpdf_p=logpdf_p,
        gram_matvec=gram_matvec,
        precondition=precondition,
    )

    # Set up value and gradient
    p_flat, unflatten = jax.flatten_util.ravel_pytree((p_prior, p_likelihood))

    def fun(p, *args):
        p_1, p_2 = unflatten(p)
        return loss(*args, params_prior=p_1, params_likelihood=p_2)

    value_and_grad = jax.value_and_grad(fun, has_aux=True)
    (value, info), grad = value_and_grad(p_flat, xs, ys, jax.random.PRNGKey(2))

    assert not jnp.isnan(value), value
    assert not jnp.any(jnp.isnan(grad)), grad
