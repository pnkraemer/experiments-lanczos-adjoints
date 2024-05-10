"""Tests for conjugate gradients."""

from typing import Callable

import jax.numpy as jnp
from matfree import test_util


def test_sth():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    solve = cg_fixed(lambda v: A @ v, len(A))
    approximation = solve(b, jnp.zeros_like(b))
    assert jnp.allclose(approximation, solution)


def cg_fixed(A: Callable, num_matvecs: int, /):
    return pcg_fixed(A, num_matvecs, lambda v: v)


def pcg_fixed(A: Callable, num_matvecs: int, M: Callable, /):
    def pcg(b, x):
        r = b - A(x)
        z = M(r)
        p = z
        for _ in range(num_matvecs):
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

    return pcg
