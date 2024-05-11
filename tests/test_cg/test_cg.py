"""Tests for conjugate gradients."""

import jax
import jax.numpy as jnp
from matfree import test_util
from matfree_extensions import cg, low_rank


def test_cg_fixed():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    solve = cg.cg_fixed_step(len(A))
    approximation, _info = solve(lambda v: A @ v, b)
    assert jnp.allclose(approximation, solution)


def test_cg_fixed_reortho():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)
    solution = jnp.linalg.solve(A, b)

    num_matvecs = len(A)
    solve = cg.cg_fixed_step_reortho(num_matvecs)
    approximation, info = solve(lambda v: A @ v, b)
    assert jnp.allclose(approximation, solution)

    num_matvecs = len(A) // 2
    solve = cg.cg_fixed_step_reortho(num_matvecs)
    _approximation, info = solve(lambda v: A @ v, b)
    Q = info["Q"]

    jnp.allclose(Q.T @ Q, jnp.eye(len(Q.T)))


def test_pcg_fixed_reortho():
    # Make a fairly ill-conditioned matrix
    sigma = 1e-6
    eta = 12.0
    eigvals = 2.0 ** (jnp.arange(-eta, eta, step=1.0))
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 1.0 + len(A))
    solution = jnp.linalg.solve(A + sigma * jnp.eye(len(A)), b)

    # Build a preconditioner and materialise
    cpp = low_rank.cholesky_partial_pivot(rank=len(A) // 2)
    precon = low_rank.preconditioner(cpp)
    pre, info = precon(lambda i, j: A[i, j], len(A))
    pre_vmapped = jax.vmap(pre, in_axes=(-1, None), out_axes=-1)
    P = pre_vmapped(jnp.eye(len(b)), sigma)
    assert info["success"]

    # Assert that PCG w/ two reorthos beats PCG w/ one reortho
    #  beats PCG w/o reortho
    num_matvecs = len(A) - 1  # exceed N to test some corner cases
    solve = cg.pcg_fixed_step(num_matvecs)
    approximation_wo, info_wo = solve(lambda v: A @ v + sigma * v, b, lambda v: P @ v)
    solve = cg.pcg_fixed_step_reortho(num_matvecs)
    approximation_w, info_w = solve(lambda v: A @ v + sigma * v, b, lambda v: P @ v)
    error_wo = jnp.amax(jnp.abs(approximation_wo - solution))
    error_w = jnp.amax(jnp.abs(approximation_w - solution))
    print(error_w, error_wo)
    assert error_w < 0.95 * error_wo
    assert False

    # Assert that PCG yields P-orthogonal vectors
    num_matvecs = len(A)
    solve = cg.pcg_fixed_step_reortho(num_matvecs)
    _approximation, info = solve(lambda v: A @ v + sigma * v, b, lambda v: P @ v)
    Q = info["Q"]
    jnp.allclose(Q.T @ P @ Q, jnp.eye(len(Q.T)))


def test_cg_fixed_num_matvecs_improves_error():
    eigvals = jnp.arange(1.0, 10.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    b = jnp.arange(1.0, 10.0)

    error = 100.0
    for n in range(len(A)):
        solve = cg.cg_fixed_step(n)
        _approximation, info = solve(lambda v: A @ v, b)

        error_now = jnp.linalg.norm(info["residual"])
        assert error_now < error, (n, error_now)
        error = error_now
