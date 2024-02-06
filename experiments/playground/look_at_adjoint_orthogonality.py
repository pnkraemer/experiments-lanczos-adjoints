"""Test the tri-diagonalisation."""

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import test_util

from matfree_extensions import lanczos

jnp.set_printoptions(3)


def random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    flat_like = jnp.arange(1.0, 1.0 + len(flat))
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


# Set up a test-matrix
n = 50
eigvals = jnp.arange(2.0, 2.0 + n)
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = _sym(matrix)

# Set up an initial vector
vector = jnp.arange(4.0, 4.0 + n)


def matvec(v, p):
    return (p + p.T) @ v


krylov_depth = 3 * n // 4
(xs, (alphas, betas)), (x, beta) = lanczos.forward(
    matvec, krylov_depth, vector, params, reortho=False
)


(dxs, (dalphas, dbetas)), (dx, dbeta) = random_like((xs, (alphas, betas)), (x, beta))

xs = jnp.concatenate([xs, x[None]])
betas = jnp.concatenate([betas, beta[None]])
dxs = jnp.concatenate([dxs, dx[None]])
dbetas = jnp.concatenate([dbetas, dbeta[None]])


# dalphas = jnp.zeros_like(dalphas)
# dbetas = jnp.zeros_like(dbetas)
# initvec = dxs[-1, :]
# initvec /= jnp.linalg.norm(initvec)
dxs = jnp.zeros_like(dxs)
# dxs = dxs.at[-1, :].set(initvec)

gradients, (lambda_0, lambda_1N, mus, nus) = lanczos.adjoint(
    matvec=matvec,
    params=(params,),
    initvec_norm=jnp.linalg.norm(vector),
    alphas=alphas,
    betas=betas,
    xs=xs,
    dalphas=dalphas,
    dbetas=dbetas,
    dxs=dxs,
    reortho=True,
)
lambdas = jnp.concatenate([lambda_0[None], lambda_1N])
print(lambdas @ xs.T)


plt.imshow(jnp.log10(jnp.abs(lambdas @ xs.T)))
plt.colorbar()
plt.show()
