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

    # Materialise the tridiagonal matrices
    dT = _dense_tridiag(dalphas, dbetas)
    T = _dense_tridiag(alphas, betas)

    # Construct basis vectors
    e_1 = jnp.eye(len(alphas))[0]
    e_K = jnp.eye(len(alphas))[-1]

    # Evaluate the constraints
    z_c = dlength.T + rho.T @ vector
    z_T = dT.T - Lt @ Qt.T
    z_r = dresidual.T - e_K.T @ (Lt - gamma * Qt)
    z_QtQ = (
        dQt @ Qt.T
        + dT.T @ T
        - T @ dT.T
        - e_1 @ (rho.T @ Qt.T)
        + M
        + gamma * e_K @ (residual @ Qt.T)
    )
    z_Q = (
        dQt
        + Lt @ matrix
        - T @ Lt
        - jnp.outer(e_1, rho)
        + M @ Qt
        + gamma * jnp.outer(e_K, residual)
    )
    return z_Q, z_r, z_T, z_c, z_QtQ


@pytest_cases.parametrize_with_cases("constraints", ".")
def test_z_Q(constraints):
    z_Q, *_ = constraints

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(z_Q)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    assert jnp.allclose(z_Q, 0.0, **tols), z_Q


@pytest_cases.parametrize_with_cases("constraints", ".")
def test_z_r(constraints):
    _, z_r, *_ = constraints

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(z_r)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    assert jnp.allclose(z_r, 0.0, **tols), z_r


@pytest_cases.parametrize_with_cases("constraints", ".")
def test_z_T(constraints):
    _, _, z_T, *_ = constraints

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(z_T)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    assert jnp.allclose(z_T, 0.0, **tols), z_T


@pytest_cases.parametrize_with_cases("constraints", ".")
def test_z_c(constraints):
    *_, z_c, _ = constraints

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(z_c)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    assert jnp.allclose(z_c, 0.0, **tols), z_c


@pytest_cases.parametrize_with_cases("constraints", ".")
def test_z_QtQ(constraints):
    *_, z_QtQ = constraints

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(z_QtQ)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    assert jnp.allclose(z_QtQ, 0.0, **tols), z_QtQ


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    key = jax.random.PRNGKey(1)
    flat_like = jax.random.uniform(key, shape=flat.shape, dtype=flat.dtype)
    flat_like = 0.1 + 0.9 * flat_like

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
