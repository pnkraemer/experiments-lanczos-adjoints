"""Test the tri-diagonalisation."""
import functools

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

(xs, (alphas, betas)), (x, beta) = lanczos.forward(
    lambda v, p: (p + p.T) @ v, 5, vector, params, reortho=True
)

print(xs.shape)
print(alphas.shape)
print(betas.shape)
print(x.shape)
print(beta.shape)

(dxs, (dalphas, dbetas)), (dx, dbeta) = random_like((xs, (alphas, betas)), (x, beta))

print(dxs.shape)
print(dalphas.shape)
print(dbetas.shape)
print(dx.shape)
print(dbeta.shape)
