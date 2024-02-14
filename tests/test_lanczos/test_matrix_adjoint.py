"""Test the matrix versions of Lanczos."""


import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos

jnp.set_printoptions(2, suppress=True)


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@pytest_cases.parametrize("krylov_depth", [5])
@pytest_cases.parametrize("n", [12])
@pytest_cases.case()
def case_constraints_mid_rank_decomposition(n, krylov_depth):
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
    w = jnp.eye(len(alphas))[-1]
    dw = jnp.zeros_like(alphas)

    dQt, (dalphas, dbetas), dresidual, dlength = _random_like(
        Qt, (alphas, betas), residual, length
    )

    _grads, (Lt, rho, M, gamma, zeta) = lanczos.matrix_adjoint(
        matvec=lambda s, p: p @ s,
        params=(matrix,),
        alphas=alphas,
        betas=betas,
        xs=Qt,
        residual=residual,
        initlength=length,
        dalphas=dalphas,
        dbetas=dbetas,
        dxs=dQt,
        dresidual=dresidual,
        dinitlength=dlength,
        w=w,
        dw=dw,
    )

    # Materialise the tridiagonal matrices
    dT = _dense_tridiag(dalphas, dbetas)
    T = _dense_tridiag(alphas, betas)

    # Construct basis vectors
    e_1 = jnp.eye(len(alphas))[0]
    e_K = jnp.eye(len(alphas))[-1]

    # Evaluate the constraints
    constraints = {
        "c": dlength.T + rho.T @ vector,
        "T": dT.T - Lt @ Qt.T,
        "w": dw.T - residual.T @ Lt.T + 2 * zeta * w.T,
        "r": dresidual.T - e_K.T @ (Lt - gamma * Qt),
        "QtQ": (
            dQt @ Qt.T
            + dT.T @ T
            - T @ dT.T
            - e_1 @ (rho.T @ Qt.T)
            + M
            + gamma * e_K @ (residual @ Qt.T)
        ),
        "Q": (
            dQt
            + Lt @ matrix
            - T @ Lt
            - jnp.outer(e_1, rho)
            + M @ Qt
            + gamma * jnp.outer(e_K, residual)
        ),
    }
    return constraints


@pytest_cases.parametrize_with_cases("constraints", ".")
@pytest_cases.parametrize("key", ["c", "T", "w", "r", "QtQ", "Q"])
def test_constraint_is_zero(constraints, key):
    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(constraints[key])).eps)
    tols = {"atol": small_value, "rtol": small_value}
    assert jnp.allclose(constraints[key], 0.0, **tols), constraints[key]


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    key = jax.random.PRNGKey(1)
    flat_like = jax.random.uniform(key, shape=flat.shape, dtype=flat.dtype)
    flat_like = 0.1 + 0.9 * flat_like

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
