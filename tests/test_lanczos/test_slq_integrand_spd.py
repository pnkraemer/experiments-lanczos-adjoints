"""Test the slq-integrand that comes with a custom VJP."""

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import hutchinson, lanczos, test_util
from matfree_extensions import lanczos as lanczos_extensions


@pytest_cases.parametrize("reortho", ["full", "none"])
@pytest_cases.parametrize("custom_vjp", [True, False])
def test_that_the_custom_vjp_matches_autodiff(reortho: str, custom_vjp: str, n=10):
    eigvals = jnp.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = jnp.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=1_000)
    order = n // 2

    # Reference
    integrand = lanczos.integrand_spd(jnp.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    key = jax.random.PRNGKey(seed=2)
    slq_value, slq_gradient = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Implementation
    kwargs = {"custom_vjp": custom_vjp, "reortho": reortho}
    integrand = lanczos_extensions.integrand_spd(jnp.log, order, matvec, **kwargs)
    estimate = hutchinson.hutchinson(integrand, sampler)
    slq_value_and_grad = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Value should be extremely close
    rtol, atol = 0.1, 0.1
    assert jnp.allclose(slq_value, slq_value_and_grad[0], rtol=rtol, atol=atol)

    # Gradients tolerances are pretty tight
    rtol, atol = 0.1, 0.1
    assert jnp.allclose(slq_gradient, slq_value_and_grad[1], rtol=rtol, atol=atol)
