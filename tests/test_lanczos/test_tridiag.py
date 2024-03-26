"""Test the tri-diagonalisation."""

import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos


def case_reortho_none():
    algorithm = lanczos.tridiag
    tols = {"atol": 1e-1, "rtol": 1e-1}
    return algorithm, tols


def case_reortho_full():
    algorithm = lanczos.tridiag_reortho_full
    tols = {"atol": 1e-5, "rtol": 1e-5}
    return algorithm, tols


@pytest_cases.parametrize_with_cases("lanczos_version", cases=".")
@pytest_cases.parametrize("ndim", [12])
def test_full_rank_reconstruction_is_exact(lanczos_version, ndim):
    # Set up a test-matrix
    eigvals = jnp.arange(1.0, 2.0, step=1 / ndim)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    impl, tols = lanczos_version
    algorithm = impl(lambda s, p: p @ s, ndim, custom_vjp=True)
    (lanczos_vectors, tridiag), _ = algorithm(vector, matrix)

    # Reconstruct the original matrix from the full-order approximation
    dense_matrix = _dense_tridiag(*tridiag)
    matrix_reconstructed = lanczos_vectors.T @ dense_matrix @ lanczos_vectors

    # Assert the reconstruction was "exact"
    assert jnp.allclose(matrix_reconstructed, matrix, **tols)

    # Assert all vectors are orthogonal
    eye = jnp.eye(len(lanczos_vectors))
    assert jnp.allclose(lanczos_vectors @ lanczos_vectors.T, eye, **tols)
    assert jnp.allclose(lanczos_vectors.T @ lanczos_vectors, eye, **tols)


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@pytest_cases.parametrize("krylov_depth", [0, 5, 11])
@pytest_cases.parametrize("ndim", [12])
@pytest_cases.parametrize("custom_vjp", [True, False])
def test_mid_rank_reconstruction_satisfies_decomposition(
    ndim, krylov_depth, custom_vjp
):
    eigvals = jnp.ones((ndim,), dtype=float) * 0.001
    eigvals_relevant = jnp.arange(1.0, 2.0, step=1 / (krylov_depth + 1))
    eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    krylov_depth = len(eigvals_relevant)
    algorithm = lanczos.tridiag(lambda s, p: p @ s, krylov_depth, custom_vjp=custom_vjp)
    (lanczos_vectors, tridiag), (q, b) = algorithm(vector, matrix)

    # Verify the decomposition
    Q = lanczos_vectors
    T = _dense_tridiag(*tridiag)
    tols = {"atol": 1e-5, "rtol": 1e-5}
    e_K = jnp.eye(krylov_depth)[-1]
    assert jnp.allclose(matrix @ Q.T, Q.T @ T + jnp.outer(e_K, q * b).T, **tols)


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
