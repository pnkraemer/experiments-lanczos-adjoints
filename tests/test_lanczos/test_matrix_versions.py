"""Test the matrix versions of Lanczos."""

import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos


def test_full_rank_reconstruction_is_exact():
    # Set up a test-matrix
    eigvals = jnp.arange(1.0, 2.0, step=0.1)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    krylov_depth=len(eigvals)
    Qt, (alphas, betas), *_ = lanczos.matrix_forward(
        lambda s, p: p @ s,
        krylov_depth,
        vector,
        matrix
    )

    # Reconstruct the original matrix from the full-order approximation
    dense_matrix = _dense_tridiag(alphas, betas)
    matrix_reconstructed = Qt.T @ dense_matrix @ Qt

    # Assert the reconstruction was "exact"
    tols = {"atol": 1e-5, "rtol": 1e-5}
    assert jnp.allclose(matrix_reconstructed, matrix, **tols)

    # Assert all vectors are orthogonal
    eye = jnp.eye(len(Qt))
    assert jnp.allclose(Qt @ Qt.T, eye, **tols)
    assert jnp.allclose(Qt.T @ Qt, eye, **tols)


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@pytest_cases.parametrize("krylov_depth", [1, 5, 11])
@pytest_cases.parametrize("n", [12])
@pytest_cases.parametrize("reortho", [True, False])
@pytest_cases.parametrize("custom_vjp", [True, False])
def test_mid_rank_reconstruction_satisfies_decomposition(
    n, krylov_depth, reortho, custom_vjp
):
    eigvals = jnp.ones((n,), dtype=float) * 0.001
    eigvals_relevant = jnp.arange(1.0, 2.0, step=1 / (krylov_depth + 1))
    eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    Qt, (alphas, betas), residual, length = lanczos.matrix_forward(
        lambda s, p: p @ s,
        krylov_depth,
        vector,
        matrix
    )

    # Verify the decomposition
    T = _dense_tridiag(alphas, betas)
    tols = {"atol": 1e-5, "rtol": 1e-5}
    e_K = jnp.eye(krylov_depth)[-1]
    assert jnp.allclose(matrix @ Qt.T, Qt.T @ T + jnp.outer(residual, e_K), **tols)


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
