"""Test the tri-diagonalisation."""

import jax.flatten_util
import jax.numpy as jnp
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
n = 20
eigvals = jnp.arange(2.0, 2.0 + n)
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = _sym(matrix)

# Set up an initial vector
vector = jnp.arange(4.0, 4.0 + n)


def matvec(v, p):
    return (p + p.T) @ v


(xs, (alphas, betas)), (x, beta) = lanczos.forward(
    matvec, n // 2, vector, params, reortho=False
)


(dxs, (dalphas, dbetas)), (dx, dbeta) = random_like((xs, (alphas, betas)), (x, beta))

xs = jnp.concatenate([xs, x[None]])
betas = jnp.concatenate([betas, beta[None]])
dxs = jnp.concatenate([dxs, dx[None]])
dbetas = jnp.concatenate([dbetas, dbeta[None]])


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
)
lambdas = jnp.concatenate([lambda_0[None], lambda_1N])
