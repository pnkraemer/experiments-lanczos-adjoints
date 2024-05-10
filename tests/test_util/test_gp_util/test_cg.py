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

    diff = Q.T @ Q - jnp.diag(jnp.diag(Q.T @ Q))
    eps = jnp.finfo(diff.dtype).eps
    assert jnp.amax(jnp.abs(diff)) < 10 * eps, diff


def test_pcg_fixed_reortho():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    # Build a preconditioner (any matrix)
    eigvals = jnp.arange(1.0, 2.0, step=1.0 / len(A))
    P = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Assert that PCG computes the correct solution
    num_matvecs = len(A)
    solve = gp_util_linalg.krylov_solve_pcg_fixed_step_reortho(num_matvecs)
    approximation, info = solve(lambda v: A @ v, b, lambda v: P @ v)
    assert jnp.allclose(approximation, solution, rtol=1e-3)

    # Assert that PCG yields P-orthogonal vectors
    num_matvecs = len(A) // 2
    solve = gp_util_linalg.krylov_solve_pcg_fixed_step_reortho(num_matvecs)
    _approximation, info = solve(lambda v: A @ v, b, lambda v: P @ v)
    Q = info["Q"]
    diff = Q.T @ P @ Q - jnp.diag(jnp.diag(Q.T @ P @ Q))
    eps = jnp.finfo(diff.dtype).eps
    assert jnp.amax(jnp.abs(diff)) < 100 * eps, diff


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
