"""Tests for conjugate gradients."""

import jax.numpy as jnp
from matfree import test_util


def test_sth():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    approximation = _cg(A, b, jnp.zeros_like(b))
    assert jnp.allclose(approximation, solution)


def _cg(A, b, x):
    return _pcg(A, b, x, jnp.eye(len(A)))


def _pcg(A, b, x, M):
    r = b - A @ x
    z = M @ r
    p = z
    for _ in range(len(A)):
        a = jnp.dot(r, z) / (p.T @ A @ p)
        x = x + a * p

        rold = r
        r = r - a * A @ p

        zold = z
        z = M @ r
        b = jnp.dot(r, z) / jnp.dot(rold, zold)
        p = z + b * p
    return x
