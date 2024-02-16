"""Test the matrix versions of Lanczos."""


import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos

jnp.set_printoptions(3, suppress=False)


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@pytest_cases.parametrize("krylov_depth", [6])
@pytest_cases.parametrize("n", [8])
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

    _grads, (Lambda, rho, Gamma, Sigma, eta) = lanczos.matrix_adjoint(
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
    return {
        "Sigma_sp": Sigma - jnp.tril(Sigma, -2),
        "Gamma_sp": Gamma - jnp.tril(Sigma),
        "c": dlength.T + rho.T @ vector,
        "T": dT.T - Lambda.T @ Qt.T + Sigma.T,
        "r": dresidual.T - e_K.T @ Lambda.T + eta.T @ Qt,
        "Q": (
            dQt
            + Lambda.T @ matrix
            - T @ Lambda.T
            - jnp.outer(e_1, rho)
            + (Gamma + Gamma.T) @ Qt
            + jnp.outer(eta, residual)
        ),
        # Projection through Q (without simplification)
        "QtQ1": (
            dQt @ Qt.T
            + Lambda.T @ matrix @ Qt.T
            - T @ dT.T
            - T @ Sigma.T
            - jnp.outer(e_1, (rho.T @ Qt.T))
            + Gamma
            + Gamma.T
        ),
        # Projection through Q (using AQ = QT + re.T)
        "QtQ2": (
            dQt @ Qt.T
            + dT.T @ T
            - T @ dT.T
            + Sigma.T @ T
            - T @ Sigma.T
            + jnp.outer(Lambda.T @ residual, e_K.T)
            - jnp.outer(e_1, (rho.T @ Qt.T))
            + Gamma
            + Gamma.T
        ),
    }


@pytest_cases.parametrize_with_cases("constraints", ".")
@pytest_cases.parametrize(
    "key", ["Sigma_sp", "Gamma_sp", "c", "T", "r", "QtQ1", "QtQ2", "Q"]
)
def test_constraint_is_zero(constraints, key):
    constraint = constraints[key]

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(constraint)).eps)

    # Assert the constraints are essentially zero
    assert jnp.allclose(constraint, 0.0, atol=small_value, rtol=small_value), constraint


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    key = jax.random.PRNGKey(1)
    flat_like = jax.random.randint(key, minval=0, maxval=2, shape=flat.shape) * 1.0

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
