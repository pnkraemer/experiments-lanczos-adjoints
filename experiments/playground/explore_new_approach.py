"""Explore."""

import jax.flatten_util
import jax.numpy as jnp
from matfree import test_util
from matfree_extensions import lanczos


def random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    # flat_like = jnp.arange(1.0, 1.0 + len(flat))
    flat_like = 0.1 + 0.9 * jax.random.uniform(jax.random.PRNGKey(1), shape=flat.shape)
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s, unflat)


# def _sym(m):
#     return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


jnp.set_printoptions(3)

# Set up a test-matrix
n = 60
eigvals = jnp.arange(2.0, 2.0 + n)
# eigvals = eigvals.at[n//2:].set(0.00001)
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = matrix


# Set up an initial vector
vector = jnp.arange(4.0, 4.0 + n)
vector /= jnp.linalg.norm(vector)


def matvec(v, p):
    return p @ v


# krylov_depth = 3 * n // 4
# krylov_depth = n // 2 + 1
krylov_depth = n // 2
# krylov_depth = n - 1

# for reortho, axis in zip([False, False], axes):
for reortho in [True]:
    (xs, (alphas, betas)), (x, beta) = lanczos.forward(
        matvec, krylov_depth, vector, params, reortho=reortho
    )

    (dxs, (dalphas, dbetas)), (dx, dbeta) = random_like(
        (xs, (alphas, betas)), (x, beta)
    )

    xs = jnp.concatenate([xs, x[None]])
    betas = jnp.concatenate([betas, beta[None]])
    dxs = jnp.concatenate([dxs, dx[None]])
    dbetas = jnp.concatenate([dbetas, dbeta[None]])

    (dv_ref, dp_ref), (lambda0, lambdas, *_) = lanczos._adjoint_pass(
        matvec=matvec,
        params=(params,),
        initvec_norm=jnp.linalg.norm(vector),
        alphas=alphas,
        betas=betas,
        xs=xs,
        dalphas=dalphas,
        dbetas=dbetas,
        dxs=dxs,
        reortho=False,
    )
    # print(lambdas.shape)
    print("Reference:", dv_ref)

    # print(dp_ref)
    # print()

    # print()

    (dv, dp), lambdas = lanczos.matrix_adjoint(
        matvec=matvec,
        params=(params,),
        initvec_norm=jnp.linalg.norm(vector),
        alphas=alphas,
        betas=betas,
        xs=xs,
        dalphas=dalphas,
        dbetas=dbetas,
        dxs=dxs,
    )
    print(dv @ vector)

    # print(dv)

    print()

    print()
    # print(xs.T @ lambdas)

    # print(lambdas @ xs.T)

    import matplotlib.pyplot as plt

    ortho = lambdas @ xs.T
    # ortho = xs.T @ lambdas
    ortho = jnp.log10(jnp.abs(ortho) + jnp.finfo(jnp.dtype(ortho)).eps)

    plt.imshow(ortho)
    plt.colorbar()
    plt.show()
    # print(lambdas[0] - vector * (vector.T @ lambdas[0]))
    # print(lambdas[0] @ vector)
    # print("product", lambdas @ xs.T)
    # print("product", lambdas.T @ xs)
