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

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))
    vector /= jnp.linalg.norm(vector)

    # Flatten the inputs
    flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))

    # Construct an identity-like function
    def eye(f):
        reconstructed = _identity(*unflatten(f), custom_vjp=custom_vjp, reortho=reortho)
        return jax.flatten_util.ravel_pytree(reconstructed)[0]

    # Assert that the function is indeed the identity
    # if n < 20:
    #     tols = {"atol": 1e-5, "rtol": 1e-5}
    #     assert jnp.allclose(eye(flat), flat, **tols)

    # Compute the Jacobian
    jacobian = jax.jit(jax.jacrev(eye))(flat)
    jacobian_reduced = _remove_zero_rows(jacobian)

    # Compute the expected Jacobian (essentially, the identity matrix)
    # The nonzero block is the Jacobian of the normalisation of the input vector
    nonzero_block = jnp.eye(len(vector)) - jnp.outer(vector, vector)
    expected = jnp.eye(len(jacobian_reduced))
    expected = expected.at[: len(vector), : len(vector)].set(nonzero_block)

    return jacobian_reduced, expected


def _identity(vector, matrix, *, custom_vjp, reortho: bool):
    """Decompose a matrix and put it back together."""
    algorithm = lanczos.tridiag(
        lambda s, p: (p + p.T) @ s, len(vector), custom_vjp=custom_vjp, reortho=reortho
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


def root_mean_square_error(x, y, /):
    # Absolute error because the target is the identity,
    # which has a lot of zeros.
    return jnp.linalg.norm(jnp.abs(x - y)) / jnp.sqrt(y.size)


if __name__ == "__main__":
    jnp.set_printoptions(2)
    for reortho in [True, False]:
        for custom_vjp in [True, False]:
            for n in [4, 12, 20, 28, 36]:
                output = evaluate_numerical_deviation_from_identity(
                    custom_vjp=custom_vjp, n=n, reortho=reortho
                )
                received, expected = output
                rmse = root_mean_square_error(received, expected)

                print()
                print(f"reortho={reortho}, custom_vjp={custom_vjp}, n={n}, rmse={rmse}")
            print()
        print()
