"""Tests for the extensions."""
import jax
from matfree import hutchinson, lanczos, test_util
from matfree.backend import np, prng

from matfree_extensions import lanczos as lanczos_extensions


def test_slq_spd_custom_vjp_matches_autodiff(n=10):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=1_000)
    order = n // 2

    # Reference
    integrand = lanczos.integrand_spd(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    key = prng.prng_key(seed=2)
    slq_value, slq_gradient = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Implementation
    integrand = lanczos_extensions.integrand_spd_custom_vjp(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    slq_value_and_grad = (jax.value_and_grad(estimate, argnums=1))(key, A)

    # Value should be extremely close
    assert np.allclose(slq_value, slq_value_and_grad[0], rtol=1e-10, atol=0.0)

    # Gradients tolerances are pretty tight
    rtol, atol = 0.1, 0.1
    assert np.allclose(slq_gradient, slq_value_and_grad[1], rtol=rtol, atol=atol)