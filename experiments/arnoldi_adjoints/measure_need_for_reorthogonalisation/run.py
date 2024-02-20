"""Measure the numerical deviation from the identity operator."""

import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
from matfree_extensions import arnoldi, exp_util


def evaluate_deviation_from_identity(*, n: int, reortho: str, custom_vjp: bool):
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

    Q, H, r, c = algorithm(vector, matrix)
    dQ, dH, dr, dc = _random_like(Q, H, r, c)

    _, multipliers = arnoldi.adjoint(
        matvec, matrix, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc, reortho=reortho
    )
    expected = multipliers["Lambda"].T @ Q - dH.T
    received = multipliers["Sigma"].T

    print("\nreceived\n", received)
    print("\nexpected\n", expected)
    print()

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(jnp.log10(jnp.finfo(c.dtype).eps + jnp.abs(expected)), vmin=-15, vmax=1)
    ax2.imshow(jnp.log10(jnp.finfo(c.dtype).eps + jnp.abs(received)), vmin=-15, vmax=1)
    plt.show()
    return multipliers["Lambda"].T @ Q - dH.T, multipliers["Sigma"].T


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    # Deterministic values because random seeds
    # would change with double precision
    # flat_like = jnp.arange(1.0, 1.0 + len(flat)) / len(flat)
    flat_like = jax.random.normal(jax.random.PRNGKey(29), shape=flat.shape)
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)


def root_mean_square_error(x, y, /):
    # Absolute error because the target is the identity,
    # which has a lot of zeros.
    return jnp.log10(jnp.linalg.norm(jnp.abs(x - y)) / jnp.sqrt(x.size))


if __name__ == "__main__":
    jnp.set_printoptions(1, suppress=False)
    intersect = "--------------------------------------------------"
    for x64 in [True]:
        jax.config.update("jax_enable_x64", x64)

        for custom_vjp_ in [True]:
            for reortho_ in ["full_with_sparsity"]:
                for n_ in jnp.arange(8, 9, step=1):
                    output = evaluate_deviation_from_identity(
                        n=n_, reortho=reortho_, custom_vjp=custom_vjp_
                    )

                    received, expected = output
                    rmse = root_mean_square_error(received, expected)

                    print(
                        f"reortho={reortho_}, custom_vjp={custom_vjp_}, "
                        f"n={n_}, rmse={rmse}"
                    )
                    print()
                print()
            print()
