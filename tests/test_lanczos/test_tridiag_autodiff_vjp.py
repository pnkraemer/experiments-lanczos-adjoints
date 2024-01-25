"""Test the tri-diagonalisation."""
import functools

import jax.flatten_util
import jax.numpy as jnp
import pytest_cases
from matfree import test_util

from matfree_extensions import lanczos


def test_vjp(n=5, krylov_order=4):
    """Test that the reverse-mode Jacobian of an identity-like operation is the identity.

    "Identity operation": full-rank Lanczos decomposition and full reconstruction.
    """
    # Set up a test-matrix
    eigvals = jax.random.uniform(jax.random.PRNGKey(2), shape=(n,)) + 1.0
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    params = _sym(matrix)

    # Set up an initial vector
    vector = jax.random.normal(jax.random.PRNGKey(1), shape=(n,))
    # todo: remove this scaling!
    vector /= jnp.linalg.norm(vector)

    # Flatten the inputs
    flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))

    # Construct an identity-like function
    def decompose(f, *, custom_vjp):
        algorithm = lanczos.tridiag(
            lambda s, p: (p + p.T) @ s, krylov_order, custom_vjp=custom_vjp
        )
        output = algorithm(*unflatten(f))
        return jax.flatten_util.ravel_pytree(output)[0]

    reference = functools.partial(decompose, custom_vjp=False)
    implementation = functools.partial(decompose, custom_vjp=True)

    fx_ref, vjp_ref = jax.vjp(reference, flat)
    fx_imp, vjp_imp = jax.vjp(implementation, flat)
    assert jnp.allclose(fx_ref, fx_imp)

    dnu = jax.random.normal(jax.random.PRNGKey(1), shape=jnp.shape(reference(flat)))
    assert jnp.allclose(*vjp_ref(dnu), *vjp_imp(dnu), atol=1e-3, rtol=1e-3)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))
