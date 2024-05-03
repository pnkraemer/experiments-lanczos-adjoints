import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
from matfree_extensions import arnoldi, exp_util


def evaluate_deviation_from_identity(*, n: int, reortho: str, custom_vjp: bool):
    # Set up a test-matrix
    matrix = exp_util.hilbert(n)
    cond = jnp.linalg.cond(matrix)
    display1 = jnp.log10(cond) + jnp.log10(jnp.finfo(matrix.dtype).eps)

    def matvec(x, p):
        # p = jnp.tril(p)  # invariance
        return p @ x

    # Set up an initial vector
    vector = jnp.ones((n,)) / jnp.sqrt(n)

    # prepare the algorithm
    krylov_depth = len(vector) // 2
    algorithm = arnoldi.arnoldi(
        matvec, krylov_depth, reortho=reortho, custom_vjp=custom_vjp
    )
    algorithm = jax.jit(algorithm)

    Q, H, r, c = algorithm(vector, matrix)
    dQ, dH, dr, dc = _random_like(Q, H, r, c)

    offdiag = jnp.diag(H, -1)

    cond = jnp.amax(offdiag) / jnp.amin(offdiag) if krylov_depth > 2 else 0.0
    display2 = jnp.log10(cond) + jnp.log10(jnp.finfo(matrix.dtype).eps)

    _, multipliers = arnoldi.adjoint(
        matvec, matrix, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc, reortho=reortho
    )

    # expected = multipliers["Lambda"].T @ Q - dH.T
    # fig, (ax1) = plt.subplots(ncols=1)
    # imshow = jnp.log10(jnp.finfo(c.dtype).eps + jnp.abs(expected))
    # colors = ax1.imshow(imshow, vmin=-14, vmax=14)
    # plt.colorbar(colors)
    # plt.show()

    orthogonality = (multipliers["Lambda"].T @ Q - dH.T, multipliers["Sigma"].T)
    conditioning = (display1, display2)
    return orthogonality, conditioning


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    # Deterministic values because random seeds
    # would change with double precision
    flat_like = jax.random.normal(jax.random.PRNGKey(29), shape=flat.shape)
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)


def root_mean_square_error(x, y, /):
    # Absolute error because the target is the identity,
    # which has a lot of zeros.
    return jnp.linalg.norm(jnp.abs(x - y)) / jnp.sqrt(x.size)


if __name__ == "__main__":
    jnp.set_printoptions(2, suppress=False)
    for x64 in [True]:
        jax.config.update("jax_enable_x64", x64)
        for custom_vjp_ in [True]:
            for reortho_ in ["full_with_sparsity", "full_without_sparsity", "none"]:
                for n_ in jnp.arange(3, 33, step=2):
                    output, (cond1, cond2) = evaluate_deviation_from_identity(
                        n=n_, reortho=reortho_, custom_vjp=custom_vjp_
                    )

                    received, expected = output
                    rmse = root_mean_square_error(received, expected)

                    print(
                        f"reortho={reortho_}, custom_vjp={custom_vjp_}, "
                        f"n={n_}, rmse={rmse}, cond={(cond1, cond2)}"
                    )
                print()
            print()
        print()
