"""Tests for the extensions."""
import jax
from matfree import hutchinson, slq, test_util
from matfree.backend import np, prng

from matfree_extensions import slq as slq_extensions


def test_slq_spd_value_and_grad_matches_autodiff(n=10):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=1_000)
    order = n // 2

    # Reference
    integrand = slq.integrand_slq_spd(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    key = prng.prng_key(seed=2)
    slq_value, slq_gradient = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Implementation
    integrand = slq_extensions.integrand_slq_spd_value_and_grad(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    slq_value_and_grad = (estimate)(key, A)

    # Value should be extremely close
    assert np.allclose(slq_value, slq_value_and_grad[0], rtol=1e-10, atol=0.0)

    # Gradients tolerances are pretty tight
    rtol, atol = 0.1, 0.1
    assert np.allclose(slq_gradient, slq_value_and_grad[1], rtol=rtol, atol=atol)


def test_slq_spd_custom_vjp_matches_autodiff(n=10):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=1_000)
    order = n // 2

    # Reference
    integrand = slq.integrand_slq_spd(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    key = prng.prng_key(seed=2)
    slq_value, slq_gradient = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Implementation
    integrand = slq_extensions.integrand_slq_spd_custom_vjp(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    slq_value_and_grad = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Value should be extremely close
    assert np.allclose(slq_value, slq_value_and_grad[0], rtol=1e-10, atol=0.0)

    # Gradients tolerances are pretty tight
    rtol, atol = 0.1, 0.1
    assert np.allclose(slq_gradient, slq_value_and_grad[1], rtol=rtol, atol=atol)


def test_hutchinson_custom_vjp_is_similar_but_different(n=3):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=10_000)
    order = n // 2

    integrand = slq.integrand_slq_spd(np.log, order, matvec)
    estimate_ref = hutchinson.hutchinson(integrand, sampler)
    estimate_custom = slq_extensions.hutchinson_custom_vjp(integrand, sampler)

    key = prng.prng_key(seed=2)
    value_custom, vjp_custom = jax.vjp(estimate_custom, key, A)
    value_ref, vjp_ref = jax.vjp(estimate_ref, key, A)

    # The forward-passes should be identical
    assert np.allclose(value_custom, value_ref)

    # The backward-passes should be different...
    vjp_custom = jax.jit(vjp_custom)
    vjp_ref = jax.jit(vjp_ref)
    assert not np.allclose(vjp_custom(1.0)[1], vjp_ref(1.0)[1])

    # ... but approximately similar
    assert np.allclose(vjp_custom(1.0)[1], vjp_ref(1.0)[1], rtol=0.25)


def test_slq_spd_custom_vjp_recursive_matches_slq_spd_custom_vjp(n=10):
    """For full orders, recursive and custom VJPs should be identical."""
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_normal(x_like, num=1)
    order = n

    reference2 = slq_extensions.integrand_slq_spd_custom_vjp(np.log, order, matvec)
    reference2 = hutchinson.hutchinson(reference2, sampler)
    implementation = slq_extensions.integrand_slq_spd_custom_vjp_rec(
        np.log, order, matvec
    )
    implementation = hutchinson.hutchinson(implementation, sampler)

    key = prng.prng_key(seed=1)
    grad_ref = jax.jit(jax.grad(reference2, argnums=1))(key, A)
    grad_impl = jax.jit(jax.grad(implementation, argnums=1))(key, A)
    assert np.allclose(grad_ref, grad_impl)
