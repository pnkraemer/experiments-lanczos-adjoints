"""Test the tri-diagonalisation."""
import functools

import jax.flatten_util
import jax.numpy as jnp
from matfree import test_util

from matfree_extensions import lanczos


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
