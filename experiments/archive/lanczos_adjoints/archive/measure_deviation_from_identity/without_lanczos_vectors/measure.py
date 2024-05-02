"""Measure the numerical deviation from the identity operator."""

import jax.flatten_util
import jax.numpy as jnp
from matfree import test_util
from matfree_extensions import lanczos


def evaluate_numerical_deviation_from_identity(
    *, custom_vjp: bool, n: int, reortho: bool
):
    # Set up a test-matrix
    eigvals = jnp.arange(1.0, 2.0, step=1 / (n + 1))
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    params = _sym(matrix)

    def matvec(s, p):
        return (p + p.T) @ s

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))
    vector /= jnp.linalg.norm(vector)

    # Compute an initial decomposition
    algorithm = lanczos.tridiag(
        matvec, len(vector), custom_vjp=custom_vjp, reortho=reortho
    )
    (lanczos_vectors, tridiag), _ = algorithm(vector, params)

    # Flatten the inputs
    flat, unflatten = jax.flatten_util.ravel_pytree(tridiag)

    def eye(t_like):
        """Put a matrix back together and decompose it again."""
        dense_matrix = _dense_tridiag(*unflatten(t_like))
        matrix_reconstructed = _sym(lanczos_vectors.T @ dense_matrix @ lanczos_vectors)
        (_vec, tridiag_), _ = algorithm(vector, matrix_reconstructed)
        return jax.flatten_util.ravel_pytree(tridiag_)[0]

    # Compute the Jacobian
    jacobian = jax.jit(jax.jacrev(eye))(flat)
    jacobian_reduced = _remove_zero_rows(jacobian)

    # Compute the expected Jacobian (the identity matrix)
    expected = jnp.eye(len(jacobian_reduced))
    return jacobian_reduced, expected


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


def _remove_zero_rows(m):
    row_is_zero = jnp.all(m == 0, axis=1)
    return m[jnp.logical_not(row_is_zero), :][:, jnp.logical_not(row_is_zero)]


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)


def root_mean_square_error(x, y, /):
    # Absolute error because the target is the identity,
    # which has a lot of zeros.
    return jnp.linalg.norm(jnp.abs(x - y)) / jnp.sqrt(y.size)


if __name__ == "__main__":
    for use_reortho in [True, False]:
        for use_custom_vjp in [True, False]:
            for nrows in [4, 12, 20, 28, 36, 44, 52]:
                # Ensure that reortho=True actually reaches the Lanczos-adjoint
                output = evaluate_numerical_deviation_from_identity(
                    custom_vjp=use_custom_vjp, n=nrows, reortho=use_reortho
                )
                received, expected = output
                rmse = root_mean_square_error(received, expected)

                print()
                print(
                    f"reortho={use_reortho}, custom_vjp={use_custom_vjp}, "
                    f"n={nrows}, rmse={rmse}"
                )

            print()
        print()
        print()
