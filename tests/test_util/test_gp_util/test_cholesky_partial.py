"""Fool around."""

import jax
import jax.numpy as jnp
from matfree import test_util

#
#
# def cholesky_pivot_partial(rank):
#     def estimate(matrix):
#
#         L = jnp.zeros((len(matrix), rank))
#
#         d = jnp.diag(matrix)
#         pi = jnp.arange(len(matrix))
#
#         for m in range(rank):
#             # Pivot (pivot!! pivot!!)
#             # (Imagine Ross Geller shouting at you)
#             i = jnp.argmax(d[pi])
#             pi_i, pi_m = pi[i], pi[m]
#             pi = pi.at[i].set(pi_m)
#             pi = pi.at[m].set(pi_i)
#
#             # Cholesky
#             lm = jnp.sqrt(d[pi[m]])
#             new = matrix[pi[m], :] - L @ L[pi[m], :] / lm
#             L = L.at[:, m].set(new)
#             L = L[pi, :]
#
#             d = d - L @ L[pi[m], :]
#
#         return L
#
#     return estimate
#


def cholesky_partial(rank):
    def estimate(matrix):
        L = jnp.zeros((len(matrix), rank))

        body = makebody(matrix)
        L = jax.lax.fori_loop(0, rank, body, L)
        return L.T

    def makebody(matrix):
        def body(i, L):
            l_ii = jnp.sqrt(matrix[i, i] - jnp.dot(L[i], L[i]))

            l_ji = matrix[:, i] - L @ L[i, :]
            l_ji /= l_ii

            return L.at[:, i].set(l_ji)

        return body

    return estimate


def test_full(n=5):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    reference = jnp.linalg.cholesky(cov).T

    cholesky_p = cholesky_partial(n)
    approximation = cholesky_p(cov)

    tol = jnp.finfo(approximation.dtype).eps
    assert jnp.allclose(approximation, reference, atol=tol, rtol=tol)


def test_partial(n=4, rank=2):
    key = jax.random.PRNGKey(1)

    cov_eig = 0.1 + jax.random.uniform(key, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    cholesky_p = cholesky_partial(rank)
    approximation = cholesky_p(cov)

    assert approximation.shape == (rank, n)
