"""Investigate whether there is any orthogonality left if dx!=0."""


import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import test_util
from matfree_extensions import lanczos
from tueplots import figsizes, fontsizes


def sym(m):
    return (m + m.T) / 2


def skew(m):
    return (m - m.T) / 2


plt.rcParams.update(figsizes.neurips2023(nrows=2, ncols=2, rel_width=1.0))
plt.rcParams.update(fontsizes.neurips2023(default_smaller=2))

# jnp.set_printoptions(2)


def random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    # flat_like = jnp.arange(1.0, 1.0 + len(flat))
    flat_like = 0.1 + 0.9 * jax.random.uniform(jax.random.PRNGKey(1), shape=flat.shape)
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


# Set up a test-matrix
n = 3
eigvals = jnp.arange(2.0, 2.0 + n)
# eigvals = eigvals.at[n//2:].set(0.00001)
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = _sym(matrix)


# Set up an initial vector
vector = jnp.arange(4.0, 4.0 + n)


def matvec(v, p):
    return (p + p.T) @ v


# krylov_depth = 3 * n // 4
# krylov_depth = n // 2 + 1
# krylov_depth = n // 2
krylov_depth = n - 1

# for reortho, axis in zip([False, False], axes):
for reortho in [True]:
    (xs, (alphas, betas)), (x, beta) = lanczos.forward(
        matvec, krylov_depth, vector, params, reortho=reortho
    )
    # print(betas)
    # print(beta)

    # todo: figure out whether max-order Krylov spaces imply
    #  that A @ x_K projected onto the Lanczos vectors is invariant
    #  and if so, whether the Krylov-projection is
    #  an invariance for the Lanczos vectors.
    (dxs, (dalphas, dbetas)), (dx, dbeta) = random_like(
        (xs, (alphas, betas)), (x, beta)
    )

    xs = jnp.concatenate([xs, x[None]])
    betas = jnp.concatenate([betas, beta[None]])
    dxs = jnp.concatenate([dxs, dx[None]])
    dbetas = jnp.concatenate([dbetas, dbeta[None]])

    # Important: reorthogonalisation only works if
    # dxs is orthogonal to xs (including if it is zero)
    # dxs1 = dxs - dxs @ xs.T @ xs
    # dxs2 = dxs @ xs.T @ xs
    # # dxs = dxs / jnp.trace(dxs)  # scale to unit(ish) values
    # print(dxs)

    # dxs2 = jnp.zeros_like(dxs)

    # print(dxs)
    # dxs =  dxs @ (xs.T @ xs)
    # dxs = xs @ sym(xs.T @ dxs)
    # dxs = xs @ skew(xs.T @ dxs) + dxs - xs @ xs.T @ dxs
    # print(dxs)

    dalphas = jnp.zeros_like(dalphas)
    dbetas = jnp.zeros_like(dbetas)
    _, (lambda_0, lambda_1N, mus, nus, xis) = lanczos._adjoint_pass(
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

    lambdas = jnp.concatenate([lambda_0[None], lambda_1N])

    print(lambdas)
    print()
    print(xs)
    print()
    print(betas)
    print(betas**2)
    print()

    print(alphas)

    # # xs = jnp.concatenate([xs, (matrix @ (xs[-1]))[None]])
    # print(xs.T @ xs)
    # print(xs @ xs.T)
    # print()
    # print(lambdas - lambdas @ xs.T @ xs)
