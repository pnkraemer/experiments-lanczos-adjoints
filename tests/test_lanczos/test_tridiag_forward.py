"""Test the tri-diagonalisation."""

import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos


@pytest_cases.parametrize("reortho", ["full", "none"])
@pytest_cases.parametrize("ndim", [12])
def test_full_rank_reconstruction_is_exact(reortho, ndim):
    # Set up a test-matrix and an initial vector
    eigvals = jnp.arange(1.0, 2.0, step=1 / ndim)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    algorithm = lanczos.tridiag(lambda s, p: p @ s, ndim, reortho=reortho)
    (lanczos_vectors, tridiag), _ = algorithm(vector, matrix)

    # Reconstruct the original matrix from the full-order approximation
    dense_matrix = _dense_tridiag(*tridiag)
    matrix_reconstructed = lanczos_vectors.T @ dense_matrix @ lanczos_vectors

    if reortho == "full":
        tols = {"atol": 1e-5, "rtol": 1e-5}
    else:
        tols = {"atol": 1e-1, "rtol": 1e-1}

    # Assert the reconstruction was "exact"
    assert jnp.allclose(matrix_reconstructed, matrix, **tols)

    # Assert all vectors are orthogonal
    eye = jnp.eye(len(lanczos_vectors))
    assert jnp.allclose(lanczos_vectors @ lanczos_vectors.T, eye, **tols)
    assert jnp.allclose(lanczos_vectors.T @ lanczos_vectors, eye, **tols)


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@pytest_cases.parametrize("krylov_depth", [1, 5, 11])
@pytest_cases.parametrize("ndim", [12])
@pytest_cases.parametrize("reortho", ["full", "none"])
def test_mid_rank_reconstruction_satisfies_decomposition(ndim, krylov_depth, reortho):
    # Set up a test-matrix and an initial vector
    eigvals = jnp.arange(1.0, 2.0, step=1 / ndim)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    algorithm = lanczos.tridiag(lambda s, p: p @ s, krylov_depth, reortho=reortho)
    (lanczos_vectors, tridiag), (q, b) = algorithm(vector, matrix)

    # Verify the decomposition
    Q, T = lanczos_vectors, _dense_tridiag(*tridiag)
    tols = {"atol": 1e-5, "rtol": 1e-5}
    e_K = jnp.eye(krylov_depth)[-1]
    assert jnp.allclose(matrix @ Q.T, Q.T @ T + jnp.outer(e_K, q * b).T, **tols)


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
