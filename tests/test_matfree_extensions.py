"""Tests for the extensions."""
from matfree import hutchinson, slq, test_util
from matfree.backend import func, np, prng

from matfree_extensions import integrand_slq_spd_with_grad


def test_slq_spd_with_grad(n=10):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=100)
    order = n // 2

    # Reference
    integrand = slq.integrand_slq_spd(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    key = prng.prng_key(seed=2)
    slq_value = func.jit(estimate)(key, A)

    # Implementation
    integrand = integrand_slq_spd_with_grad(np.log, order, matvec)
    estimate = hutchinson.hutchinson(integrand, sampler)
    slq_value_and_grad = func.jit(estimate)(key, A)

    # Value should be extremely close
    assert np.allclose(slq_value, slq_value_and_grad["value"], rtol=1e-10, atol=0.0)
