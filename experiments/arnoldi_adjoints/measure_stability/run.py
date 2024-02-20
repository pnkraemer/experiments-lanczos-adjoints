"""Measure the numerical deviation from the identity operator."""

import jax.flatten_util
import jax.numpy as jnp
from matfree_extensions import arnoldi, exp_util


def evaluate_deviation_from_identity(*, n: int, reortho: str, custom_vjp: bool):
    def _syml(m):
        return jnp.tril(m) - 0.5 * jnp.diag(jnp.diag(m))

    # Set up a test-matrix
    # matrix = _syml(exp_util.hilbert(n))
    matrix = exp_util.hilbert(n)
    cond = jnp.linalg.cond(exp_util.hilbert(n))
    display = jnp.log10(cond) + jnp.log10(jnp.finfo(matrix.dtype).eps)

    def matvec(x, p):
        # p = jnp.tril(p)  # invariance
        return p @ x

    # Set up an initial vector
    vector = jnp.ones((n,)) / jnp.sqrt(n)

    # prepare the algorithm
    algorithm = arnoldi.arnoldi(
        matvec, len(vector), reortho=reortho, custom_vjp=custom_vjp
    )
    algorithm = jax.jit(algorithm)

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
    # jacobian = _remove_zero_rows(jacobian)
    jacobian = (jacobian + jacobian.T) / 2

    # Measure deviation (for reference)
    Q, H, *_ = algorithm(vector, matrix)

    return (jacobian, jnp.eye(len(jacobian))), ((Q @ H @ Q.T) - matrix, 0.0), display


def _remove_zero_rows(m):
    row_is_zero = jnp.all(m == 0, axis=1)
    return m[jnp.logical_not(row_is_zero), :][:, jnp.logical_not(row_is_zero)]


def root_mean_square_error(x, y, /):
    # Absolute error because the target is the identity,
    # which has a lot of zeros.
    return jnp.log10(jnp.linalg.norm(jnp.abs(x - y)) / jnp.sqrt(x.size))


if __name__ == "__main__":
    jnp.set_printoptions(2, suppress=False)
    intersect = "--------------------------------------------------"
    for x64 in [False]:
        jax.config.update("jax_enable_x64", x64)

        for custom_vjp_ in [True]:
            for reortho_ in ["full_with_sparsity", "full_without_sparsity", "none"]:
                for n_ in jnp.arange(2, 20, step=2):
                    output, reconstruction, cond = evaluate_deviation_from_identity(
                        n=n_, reortho=reortho_, custom_vjp=custom_vjp_
                    )

                    received, expected = output
                    rmse = root_mean_square_error(received, expected)

                    received, expected = reconstruction
                    rmse__ = root_mean_square_error(received, expected)

                    print(
                        f"reortho={reortho_}, custom_vjp={custom_vjp_}, "
                        f"n={n_}, rmse={rmse}, recon={rmse__}, cond={cond}"
                    )
                    print()
                print()
            print()
