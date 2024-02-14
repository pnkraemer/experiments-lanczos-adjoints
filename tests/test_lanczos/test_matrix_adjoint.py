"""Test the matrix versions of Lanczos."""


import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@pytest_cases.parametrize("krylov_depth", [5])
@pytest_cases.parametrize("n", [12])
def test_mid_rank_reconstruction_satisfies_decomposition(n, krylov_depth):
    eigvals = jnp.ones((n,), dtype=float) * 0.001
    eigvals_relevant = jnp.arange(1.0, 2.0, step=1 / krylov_depth)
    eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    # Run Lanczos approximation
    Qt, (alphas, betas), residual, length = lanczos.matrix_forward(
        lambda s, p: p @ s, krylov_depth, vector, matrix
    )
    dQt, (dalphas, dbetas), dresidual, dlength = _random_like(
        Qt, (alphas, betas), residual, length
    )

    _grads, (Lt, rho, M, gamma) = lanczos.matrix_adjoint(
        matvec=lambda s, p: p @ s,
        params=(matrix,),
        alphas=alphas,
        betas=betas,
        xs=Qt,
        res=residual,
        norm=length,
        dalphas=dalphas,
        dbetas=dbetas,
        dxs=dQt,
        dres=dresidual,
        dnorm=dlength,
    )

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(T)).eps)
    tols = {"atol": small_value, "rtol": small_value}
    e_K = jnp.eye(krylov_depth)[-1]

    # Materialise the tridiagonal matrices
    dT = _dense_tridiag(dalphas, dbetas)
    T = _dense_tridiag(alphas, betas)

    # Evaluate the constraints
    z_c = dlength.T + rho.T @ vector
    z_T = dT.T - Lt @ Qt.T
    z_r = dresidual.T - e_K.T @ (Lt + gamma * Qt)
    z_Q = (
        dQ.T
        + Lt @ matrix
        - T @ Lt
        - jnp.outer(e1, rho)
        + M @ Qt
        + gamma * jnp.outer(e_K, residual)
    )

    # Test the constraints are all zero
    assert jnp.allclose(z_c, 0.0, **tols)
    assert jnp.allclose(z_T, 0.0, **tols)
    assert jnp.allclose(z_r, 0.0, **tols)
    assert jnp.allclose(z_Q, 0.0, **tols)


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    key = jax.random.PRNGKey(1)
    flat_like = jax.random.uniform(key, shape=flat.shape, dtype=flat.dtype)
    flat_like = 0.1 + 0.9 * flat_like

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
