"""Invert a triangular matrix."""

import jax
import jax.numpy as jnp


def random_triu(seed, n):
    key = jax.random.PRNGKey(seed)
    return jnp.triu(jax.random.normal(key, shape=(n, n)))


def solve(LHS, RHS):
    X = jnp.zeros_like(RHS)

    # First row
    for i in range(len(LHS) - 1, -1, -1):
        X = X.at[i, :].set((RHS[i, :] - LHS[i, :] @ X) / LHS[i, i])

    return X


nrows = 5
A, B = (random_triu(i, nrows) for i in range(2))

X_ref = jnp.linalg.solve(A, B)
X_cmp = solve(A, B)
assert jnp.allclose(X_ref, X_cmp)
