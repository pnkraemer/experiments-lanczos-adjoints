"""Tests for conjugate gradients."""

import jax.numpy as jnp
from matfree import test_util
from matfree_extensions.util import gp_util_linalg


def test_cg_fixed():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    solve = gp_util_linalg.krylov_solve_cg_fixed(len(A))
    approximation, _info = solve(lambda v: A @ v, b)
    assert jnp.allclose(approximation, solution)


def test_cg_fixed_num_matvecs_improves_error():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)

    error = 100.0
    for n in range(len(A)):
        solve = gp_util_linalg.krylov_solve_cg_fixed(n)
        _approximation, info = solve(lambda v: A @ v, b)

        error_now = jnp.linalg.norm(info["residual"])
        assert error_now < error, (n, error_now)
        error = error_now
