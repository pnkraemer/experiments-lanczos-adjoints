"""Tests for log-marginal likelihoods."""

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import hutchinson
from matfree_extensions.util import gp_util, gp_util_linalg


@pytest_cases.parametrize(
    "low_rank",
    [gp_util_linalg.low_rank_cholesky_pivot, gp_util_linalg.low_rank_cholesky],
)
def test_preconditioning_reduces_cg_iteration_count(
    low_rank, n=100, rank=5, tol=1e-4, max_steps=100
):
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
    solve = gp_util_linalg.krylov_solve_cg_lineax(
        atol=tol, rtol=tol, max_steps=max_steps
    )
    logpdf = gp_util.logpdf_krylov(solve=solve, logdet=logdet)
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf, gram_matvec=gram_matvec)

    # Precon:
    solve_p = gp_util_linalg.krylov_solve_cg_lineax_precondition(
        atol=tol, rtol=tol, max_steps=max_steps
    )
    logpdf_p = gp_util.logpdf_krylov_precondition(solve_p=solve_p, logdet=logdet)
    low_rank_impl = low_rank(n, rank=rank)
    precon = gp_util_linalg.precondition_low_rank(low_rank_impl, small_value=1e-4)
    loss_p = gp_util.mll_exact_precondition(
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

    # Compare values
    assert info_p["num_steps"] < info["num_steps"]
    assert jnp.allclose(value_p, value, atol=1e-1, rtol=1e-1)


@pytest_cases.parametrize(
    "low_rank",
    [gp_util_linalg.low_rank_cholesky_pivot, gp_util_linalg.low_rank_cholesky],
)
def test_preconditioning_is_differentiable(
    low_rank, n=100, rank=2, tol=1e-4, maxiter=100
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
    solve_p = gp_util_linalg.krylov_solve_cg_precondition(tol=tol, maxiter=maxiter)
    gram_matvec = gp_util_linalg.gram_matvec_full_batch()
    logpdf_p = gp_util.logpdf_krylov_precondition(solve_p=solve_p, logdet=logdet)

    # Set up an MLL
    low_rank_impl = low_rank(n, rank=rank)
    precondition = gp_util_linalg.precondition_low_rank(low_rank_impl, small_value=1e-4)
    loss = gp_util.mll_exact_precondition(
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


#
# @pytest_cases.parametrize(
#     "low_rank", [gp_util.low_rank_cholesky_pivot, gp_util.low_rank_cholesky]
# )
# def test_preconditioning_small_value_does_not_affect_the_solution(
#     low_rank, n=100, rank=5, tol=1e-4, maxiter=100
# ):
#     # Set up a testproblem
#     xs = jnp.linspace(0, 1, num=n)
#     ys = jnp.linspace(0, 1, num=n)
#     k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
#     likelihood, p_likelihood = gp_util.likelihood_gaussian()
#
#     # Set up all constant parameters
#     gram_matvec = gp_util.gram_matvec_full_batch()
#     v_like = xs
#     sample = hutchinson.sampler_normal(v_like, num=10)
#     logdet = gp_util.krylov_logdet_slq(5, sample=sample, num_batches=1)
#
#     # [PRECON] Set up a solver
#     solve = gp_util.krylov_solve_cg_precondition(tol=tol, maxiter=maxiter)
#     logpdf_fun = gp_util.logpdf_krylov(solve=solve, logdet=logdet)
#
#     # [PRECON] Set up a GP model
#     low_rank_impl = low_rank(n, rank=rank)
#     P = gp_util.precondition_low_rank(low_rank_impl, small_value=1e-4)
#     prior = gp_util.model_precondition(
#         gp_util.mean_zero(), k, gram_matvec=gram_matvec, precondition=P
#     )
#     loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf_fun)
#
#     # Set up value and gradient
#     p_flat, unflatten = jax.flatten_util.ravel_pytree((p_prior, p_likelihood))
#
#     def fun(p, *args):
#         p_1, p_2 = unflatten(p)
#         return loss(*args, params_prior=p_1, params_likelihood=p_2)
#
#     value_and_grad = jax.value_and_grad(fun, has_aux=True)
#     (value, info), grad = value_and_grad(p_flat, xs, ys, jax.random.PRNGKey(2))
#
#     print(value)
#     print(info)
#     print(grad)
#     assert False
