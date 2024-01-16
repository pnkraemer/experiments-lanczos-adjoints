"""Tests for the extensions."""
import jax
from matfree import hutchinson, lanczos, test_util
from matfree.backend import np, prng

from matfree_extensions import hutchinson as hutchinson_extensions


def test_custom_vjp_is_similar_but_different(n=3):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=10_000)
    order = n // 2

    integrand = lanczos.integrand_spd(np.log, order, matvec)
    estimate_ref = hutchinson.hutchinson(integrand, sampler)
    estimate_custom = hutchinson_extensions.hutchinson_custom_vjp(integrand, sampler)

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
