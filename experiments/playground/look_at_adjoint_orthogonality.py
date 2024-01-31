"""Test the tri-diagonalisation."""

import jax.flatten_util
import jax.numpy as jnp
from matfree import test_util

from matfree_extensions import lanczos


def random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    flat_like = jax.random.normal(
        jax.random.PRNGKey(14214213), shape=flat.shape, dtype=flat.dtype
    )
    return unflatten(flat_like)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


# Set up a test-matrix
n = 10
eigvals = jax.random.uniform(jax.random.PRNGKey(2), shape=(n,)) + 1.0
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = _sym(matrix)

# Set up an initial vector
vector = jax.random.normal(jax.random.PRNGKey(1), shape=(n,))


def matvec(v, p):
    return (p + p.T) @ v


(xs, (alphas, betas)), (x, beta) = lanczos.forward(
    matvec, 5, vector, params, reortho=True
)


(dxs, (dalphas, dbetas)), (dx, dbeta) = random_like((xs, (alphas, betas)), (x, beta))

xs = jnp.concatenate([xs, x[None]])
betas = jnp.concatenate([betas, beta[None]])
dxs = jnp.concatenate([dxs, dx[None]])
dbetas = jnp.concatenate([dbetas, dbeta[None]])


gradients, (lambda_0, lambda_1N) = lanczos.adjoint(
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
print(lambdas.shape)
