"""Test the tri-diagonalisation."""
import jax.numpy as jnp
from matfree import test_util

from matfree_extensions import lanczos


def test_full_rank_reconstruction_is_exact():
    # Set up a test-matrix
    eigvals = jnp.arange(1.0, 2.0, step=0.1)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    algorithm = lanczos.tridiag(lambda s, p: p @ s, len(vector))
    (lanczos_vectors, tridiag), _ = algorithm(vector, matrix)

    # Reconstruct the original matrix from the full-order approximation
    dense_matrix = _dense_tridiag(*tridiag)
    matrix_reconstructed = lanczos_vectors.T @ dense_matrix @ lanczos_vectors

    # Assert the reconstruction was "exact"
    tols = {"atol": 1e-5, "rtol": 1e-5}
    assert jnp.allclose(matrix_reconstructed, matrix, **tols)

    # Assert all vectors are orthogonal
    eye = jnp.eye(len(lanczos_vectors))
    assert jnp.allclose(lanczos_vectors @ lanczos_vectors.T, eye, **tols)
    assert jnp.allclose(lanczos_vectors.T @ lanczos_vectors, eye, **tols)


def test_mid_rank_reconstruction_satisfies_decomposition():
    pass


def test_zero_rank_reconstruction_does_not_fail():
    pass


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
