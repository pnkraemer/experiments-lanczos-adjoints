"""Measure the numerical deviation from the identity operator."""

import jax.flatten_util
import jax.numpy as jnp
from matfree_extensions import arnoldi, exp_util


def evaluate_numerical_deviation_from_identity(
    *, n: int, reortho: bool, custom_vjp: bool
):
    # Set up a test-matrix
    matrix = exp_util.hilbert(n)

    def matvec(x, p):
        return p @ x

    # Set up an initial vector
    vector = jnp.ones((n,)) / jnp.sqrt(n)

    # prepare the algorithm
    algorithm = arnoldi.arnoldi(
        matvec, len(vector), reortho=reortho, custom_vjp=custom_vjp
    )

    # Flatten the inputs
    flat, unflatten = jax.flatten_util.ravel_pytree(matrix)

    # Construct an identity-like function
    def eye(f):
        mat = unflatten(f)
        Q, H, r, c = algorithm(vector, mat)
        mat = Q @ H @ Q.T
        return jax.flatten_util.ravel_pytree(mat)[0]

    # Compute the Jacobian
    jacobian = jax.jit(jax.jacrev(eye))(flat)
    jacobian = (jacobian + jacobian.T) / 2

    # Measure loss of orthogonality (for reference)
    Q, *_ = algorithm(vector, matrix)

    # Measure deviation
    Q, H, *_ = algorithm(vector, matrix)

    return (
        (jacobian, jnp.eye(len(jacobian))),
        (Q.T @ Q, jnp.eye(len(Q))),
        (Q @ H @ Q.T - matrix, 0.0),
    )


def root_mean_square_error(x, y, /):
    # Absolute error because the target is the identity,
    # which has a lot of zeros.
    return jnp.linalg.norm(jnp.abs(x - y)) / jnp.sqrt(x.size)


if __name__ == "__main__":
    jnp.set_printoptions(7, suppress=True)

    for custom_vjp_ in [True, False]:
        for reortho_ in [True, False]:
            for n_ in jnp.arange(1, 10, step=1):
                (
                    output,
                    orthogonality,
                    reconstruction,
                ) = evaluate_numerical_deviation_from_identity(
                    n=n_, reortho=reortho_, custom_vjp=custom_vjp_
                )
                received, expected = output
                rmse = root_mean_square_error(received, expected)

                received, expected = orthogonality
                rmse_ = root_mean_square_error(received, expected)

                received, expected = reconstruction
                rmse__ = root_mean_square_error(received, expected)

                print(
                    f"reortho={reortho_}, custom_vjp={custom_vjp_}, "
                    f"n={n_}, rmse={rmse}, ortho={rmse_}, recon={rmse__}"
                )
                print()
            print()
