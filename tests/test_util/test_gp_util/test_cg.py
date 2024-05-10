"""Tests for conjugate gradients."""

from typing import Callable

import jax
import jax.numpy as jnp
from matfree import test_util


def test_cg_fixed():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    solve = cg_fixed(len(A))
    approximation, _info = solve(lambda v: A @ v, b)
    assert jnp.allclose(approximation, solution)


def test_cg_fixed_num_matvecs_improves_error():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)

    error = 100.0
    for n in range(len(A)):
        solve = cg_fixed(n)
        _approximation, info = solve(lambda v: A @ v, b)

        error_now = jnp.linalg.norm(info["residual"])
        assert error_now < error, (n, error_now)
        error = error_now


def cg_fixed(num_matvecs: int, /):
    return pcg_fixed(num_matvecs, lambda v: v)


def pcg_fixed(num_matvecs: int, M: Callable, /):
    def pcg(A: Callable, b: jax.Array):
        return jax.lax.custom_linear_solve(A, b, pcg_impl, symmetric=True, has_aux=True)

    def pcg_impl(A: Callable, b):
        x = jnp.zeros_like(b)

        r = b - A(x)
        z = M(r)
        p = z

        body_fun = make_body(A)
        init = (x, p, r, z)
        x, p, r, z = jax.lax.fori_loop(0, num_matvecs, body_fun, init_val=init)
        return x, {"residual": r}

    def make_body(A):
        def body_fun(_i, state):
            x, p, r, z = state
            Ap = A(p)
            a = jnp.dot(r, z) / (p.T @ Ap)
            x = x + a * p

            rold = r
            r = r - a * Ap

            zold = z
            z = M(r)
            b = jnp.dot(r, z) / jnp.dot(rold, zold)
            p = z + b * p
            return x, p, r, z

        return body_fun

    return pcg
