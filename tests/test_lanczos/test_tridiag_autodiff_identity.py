"""Test the tri-diagonalisation."""
import jax.flatten_util
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos


@pytest_cases.parametrize("custom_vjp", [True, False])
def test_identity_operator(custom_vjp, n=5):
    """Test that the reverse-mode Jacobian of an identity operation is the identity.

    "Identity operation": full-rank Lanczos decomposition and full reconstruction.
    """
    # Set up a test-matrix
    eigvals = jnp.arange(1.0, 2.0, step=1 / (n + 1))
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    params = _sym(matrix)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))
    vector /= jnp.linalg.norm(vector)

    # Flatten the inputs
    flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))

    # Construct an identity-like function
    def eye(f):
        reconstructed = _identity(*unflatten(f), custom_vjp=custom_vjp)
        return jax.flatten_util.ravel_pytree(reconstructed)[0]

    # Assert that the function is indeed the identity
    tols = {"atol": 1e-5, "rtol": 1e-5}
    assert jnp.allclose(eye(flat), flat, **tols)

    # Compute the Jacobian
    jacobian = jax.jit(jax.jacrev(eye))(flat)
    jacobian_reduced = _remove_zero_rows(jacobian)

    # Compute the expected Jacobian (essentially, the identity matrix)
    # The nonzero block is the Jacobian of the normalisation of the input vector
    nonzero_block = jnp.eye(len(vector)) - jnp.outer(vector, vector)
    expected = jnp.eye(len(jacobian_reduced))
    expected = expected.at[: len(vector), : len(vector)].set(nonzero_block)

    # Assert that the Jacobian has the expected form.
    tols = {"atol": 1e-3, "rtol": 1e-3}
    assert jnp.allclose(jacobian_reduced, expected, **tols)


def _identity(vector, matrix, *, custom_vjp):
    algorithm = lanczos.tridiag(
        lambda s, p: (p + p.T) @ s, len(vector), custom_vjp=custom_vjp
    )
    (lanczos_vectors, tridiag), _ = algorithm(vector, matrix)

    # Reconstruct the original matrix from the full-order approximation
    dense_matrix = _dense_tridiag(*tridiag)
    matrix_reconstructed = lanczos_vectors.T @ dense_matrix @ lanczos_vectors

    # Reconstruct the original matrix
    vector_reconstructed = lanczos_vectors[0, :]

    # Return the reconstruction
    return vector_reconstructed, _sym(matrix_reconstructed)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


def _remove_zero_rows(m):
    row_is_zero = jnp.all(m == 0, axis=1)
    return m[jnp.logical_not(row_is_zero), :][:, jnp.logical_not(row_is_zero)]


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
