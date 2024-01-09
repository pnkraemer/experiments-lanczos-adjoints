"""Tests for the extensions."""
import jax
import jax.numpy as jnp
from matfree import hutchinson, slq, test_util
from matfree.backend import np, prng

from matfree_extensions import slq as slq_extensions


def test_integrand_slq_spd_value_and_grad(n=10):
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


def test_integrand_slq_spd_custom_vjp(n=10):
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


def test_hutchinson_custom_vjp(n=3):
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
    # print(vjp_custom(1.0)[1])
    # print(vjp_ref(1.0)[1])
    assert np.allclose(vjp_custom(1.0)[1], vjp_ref(1.0)[1], rtol=0.25)


def test_integrand_slq_spd_custom_vjp_recursive(n=3):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    def logdet(p):
        return jnp.linalg.slogdet(p)[1]

    truth = jax.value_and_grad(logdet)(A)
    # print(truth)
    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_normal(x_like, num=40)
    order = n

    reference1 = slq.integrand_slq_spd(np.log, order, matvec)
    reference2 = slq_extensions.integrand_slq_spd_custom_vjp(np.log, order, matvec)
    reference2 = hutchinson.hutchinson(reference2, sampler)
    implementation = slq_extensions.integrand_slq_spd_custom_vjp_rec(
        np.log, order, matvec
    )
    implementation = hutchinson.hutchinson(implementation, sampler)

    key = prng.prng_key(seed=1)
    # x = sampler(key)[0]
    # print(jax.value_and_grad(reference1, argnums=(0, 1))(x, A))
    print(jax.grad(reference2, argnums=1)(key, A))
    print(jax.grad(implementation, argnums=1)(key, A))

    # print(jax.grad(reference2, argnums=1)(x, A) / jax.grad(implementation, argnums=1)(x, A))

    # estimate = hutchinson.hutchinson(integrand, sampler)
    # slq_value, slq_gradient = (jax.value_and_grad(estimate, argnums=1))(key, A)
    # print(slq_value)
    # print(slq_gradient)
    # print()
    # estimate = hutchinson.hutchinson(integrand, sampler)
    # # slq_value = estimate(key, A)
    # slq_value_and_grad = (jax.value_and_grad(estimate, argnums=1))(key, A)
    # print(slq_value_and_grad[0])
    # print(slq_value_and_grad[1])

    assert False

    # Value should be extremely close
    assert np.allclose(slq_value, slq_value_and_grad[0], rtol=1e-10, atol=0.0)

    # Gradients tolerances are pretty tight
    rtol, atol = 0.1, 0.1
    assert np.allclose(slq_gradient, slq_value_and_grad[1], rtol=rtol, atol=atol)


def test_asymmetric_lanczos_product(n=4):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    def matvec(x, p):
        return p @ x

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=1)
    order = n

    key = prng.prng_key(seed=2)
    x = sampler(key)[0]
    y = matvec(x, A)

    # Expected
    eigvals, eigvecs = jnp.linalg.eigh(A)
    logA = eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T
    expected = x.T @ logA @ y

    # Received via Lanczos
    reference = slq_extensions.integrand_slq_spd(lambda s: np.log(s) * s, order, matvec)
    received_via_lanczos = reference(x, A)

    # Received via Trick
    implementation = slq_extensions.integrand_slq_spd(
        lambda s: np.log(s), order, matvec
    )
    received_via_trick = (implementation(x + y, A) - implementation(x - y, A)) / 4

    assert np.allclose(received_via_lanczos, expected)
    assert np.allclose(received_via_trick, expected)

    # print(x.T @ logm @ y + y.T @ logm @ x)
    # print(logm @ A)
    assert False


def test_asymmetric_product(n=4):
    eigvals = np.arange(0.0, 1.0 + n, step=1.0) + 1.0
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    x_like = np.ones((len(A),))
    sampler = hutchinson.sampler_rademacher(x_like, num=2)

    key = prng.prng_key(seed=2)
    x, y = sampler(key)
    y = A @ x
    eigvals, eigvecs = jnp.linalg.eigh(A)
    logA = eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T

    expected = x.T @ logA @ y
    received = ((x + y).T @ logA @ (x + y) - (x - y).T @ logA @ (x - y)) / 4

    assert np.allclose(expected, received)
    print(expected)
    print(received)
    assert False
