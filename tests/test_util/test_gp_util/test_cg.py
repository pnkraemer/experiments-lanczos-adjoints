"""Tests for conjugate gradients."""

import jax.numpy as jnp
from matfree import test_util
from matfree_extensions.util import gp_util_linalg


def test_cg_fixed():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    solve = gp_util_linalg.krylov_solve_cg_fixed_step(len(A))
    approximation, _info = solve(lambda v: A @ v, b)
    assert jnp.allclose(approximation, solution)


def test_cg_fixed_reortho():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    num_matvecs = len(A)
    solve = gp_util_linalg.krylov_solve_cg_fixed_step_reortho(num_matvecs)
    approximation, info = solve(lambda v: A @ v, b)
    assert jnp.allclose(approximation, solution)

    num_matvecs = len(A) // 2
    solve = gp_util_linalg.krylov_solve_cg_fixed_step_reortho(num_matvecs)
    _approximation, info = solve(lambda v: A @ v, b)
    Q = info["Q"]
    assert jnp.allclose(Q.T @ Q, jnp.eye(num_matvecs), atol=1e-4, rtol=1e-4)


def test_cg_fixed_num_matvecs_improves_error():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)

    error = 100.0
    for n in range(len(A)):
        solve = gp_util_linalg.krylov_solve_cg_fixed_step(n)
        _approximation, info = solve(lambda v: A @ v, b)

        error_now = jnp.linalg.norm(info["residual"])
        assert error_now < error, (n, error_now)
        error = error_now
