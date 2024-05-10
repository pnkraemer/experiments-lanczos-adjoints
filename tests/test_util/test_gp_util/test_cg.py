"""Tests for conjugate gradients."""

from typing import Callable

import jax.numpy as jnp
from matfree import test_util


def test_sth():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    approximation = _cg(lambda v: A @ v, b, jnp.zeros_like(b))
    assert jnp.allclose(approximation, solution)


def _cg(A: Callable, b, x):
    return _pcg(A, b, x, lambda v: v)


def _pcg(A: Callable, b, x, M: Callable):
    r = b - A(x)
    z = M(r)
    p = z
    for _ in range(len(x)):
        Ap = A(p)
        a = jnp.dot(r, z) / (p.T @ Ap)
        x = x + a * p

        rold = r
        r = r - a * Ap

        zold = z
        z = M(r)
        b = jnp.dot(r, z) / jnp.dot(rold, zold)
        p = z + b * p
    return x
