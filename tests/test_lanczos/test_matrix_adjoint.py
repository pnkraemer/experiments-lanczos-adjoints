"""Test the matrix versions of Lanczos."""


import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions import lanczos

jnp.set_printoptions(3, suppress=False)


@pytest_cases.parametrize("krylov_depth", [3])
@pytest_cases.parametrize("n", [7])
@pytest_cases.case()
def test_vjp(n, krylov_depth):
    eigvals = jnp.arange(1.0, 1.0 + n, step=1.0)
    matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    # Set up an initial vector
    vector = jnp.flip(jnp.arange(1.0, 1.0 + len(eigvals)))

    def fwd(v, mat):
        # Run Lanczos approximation
        return lanczos.matrix_forward(lambda s, p: p @ s, krylov_depth, v, mat)

    (Qt, (alphas, betas), residual, length), vjp = jax.vjp(fwd, vector, matrix)

    dQt, (dalphas, dbetas), dresidual, dlength = _random_like(
        Qt, (alphas, betas), residual, length
    )

    dalphas = jnp.zeros_like(dalphas)
    dbetas = jnp.zeros_like(dbetas)
    dQt = jnp.zeros_like(dQt)
    # dresidual = jnp.zeros_like(dresidual)
    dlength = jnp.zeros_like(dlength)

    dv_ref, dA_ref = vjp((dQt, (dalphas, dbetas), dresidual, dlength))

    (dv, dA), (Lambda, lambda_, Gamma, Sigma, eta) = lanczos.matrix_adjoint(
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
    )
    print(Lambda)
    print(lambda_)
    print(Gamma)
    print(Sigma)
    print(eta)

    print()
    print("----------------------------------------")
    print("----------------------------------------")
    print("ref", dA_ref)
    print()
    print(dA)
    print()
    print()
    print()
    print()
    print("ref", dv_ref)
    print()
    print(dv)
    print()
    print()

    assert False


# anything 0 <= k < n works; k=n is full reconstruction
# and the (q, b) values become meaningless
@pytest_cases.parametrize("krylov_depth", [6])
@pytest_cases.parametrize("n", [8])
@pytest_cases.case()
def case_constraints_mid_rank_decomposition(n, krylov_depth):
    eigvals = jnp.arange(1.0, 1.0 + n, step=1.0)

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
    )

    # Materialise the tridiagonal matrices
    dT = _dense_tridiag(dalphas, dbetas)
    T = _dense_tridiag(alphas, betas)

    # Construct basis vectors
    e_1 = jnp.eye(len(alphas))[0]
    e_K = jnp.eye(len(alphas))[-1]

    # Evaluate the constraints
    constraints = {
        "Sigma_sp": Sigma - jnp.tril(Sigma, -2),
        "Gamma_sp": Gamma - jnp.tril(Gamma),
        "c": dlength.T + rho.T @ vector,
        "T": (dT.T - Lambda.T @ Qt.T + Sigma.T),
        "r": dresidual.T - e_K.T @ Lambda.T + eta.T @ Qt,
        "Q": (
            dQt
            + Lambda.T @ matrix
            - T @ Lambda.T
            - jnp.outer(e_1, rho)
            + (Gamma + Gamma.T) @ Qt
            + jnp.outer(eta, residual)
        ),
        # # Projection through Q (without simplification)
        "QtQ": (
            dQt @ Qt.T
            + Lambda.T @ matrix @ Qt.T
            - T @ dT.T
            - T @ Sigma.T
            - jnp.outer(e_1, (rho.T @ Qt.T))
            + Gamma
            + Gamma.T
        ),
    }
    return constraints


@pytest_cases.parametrize_with_cases("constraints", ".")
@pytest_cases.parametrize("key", ["Sigma_sp", "Gamma_sp", "c", "T", "r", "Q", "QtQ"])
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
    flat_like = 0.1 + jax.random.uniform(key, shape=flat.shape, dtype=flat.dtype)

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
