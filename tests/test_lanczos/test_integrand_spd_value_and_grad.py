"""Test the slq-integrand that comes with a custom VJP."""

import jax
import jax.numpy as jnp
from matfree import hutchinson, lanczos, test_util
from matfree_extensions import lanczos as lanczos_extensions


def test_integrand_spd_value_and_grad_match_matfree_plus_autodiff(n=10):
    eigvals = jnp.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    A = _sym(A)

    def matvec(x, p):
        return (p + p.T) @ x

    x_like = jnp.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=1)
    order = n // 2

    # Reference
    integrand = lanczos.integrand_spd(jnp.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    key = jax.random.PRNGKey(seed=2)
    slq_value, slq_gradient = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Implementation
    integrand = lanczos_extensions.integrand_spd(jnp.log, order + 1, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    slq_value_and_grad = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Assert the similarity
    small_value = jnp.sqrt(jnp.finfo(slq_value.dtype).eps)
    assert jnp.allclose(slq_value, slq_value_and_grad[0], rtol=small_value)
    assert jnp.allclose(slq_gradient, slq_value_and_grad[1], rtol=small_value)


def test_integrand_spd_value_and_grad_match_matfree_plus_autodiff_reuse_lanczos(n=10):
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
    integrand = lanczos_extensions.integrand_spd(jnp.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    slq_value_and_grad = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Value should be extremely close
    rtol, atol = 0.1, 0.1
    assert jnp.allclose(slq_value, slq_value_and_grad[0], rtol=rtol, atol=atol)

    # Gradients tolerances are pretty loose
    rtol, atol = 0.1, 0.1
    assert jnp.allclose(slq_gradient, slq_value_and_grad[1], rtol=rtol, atol=atol)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))
