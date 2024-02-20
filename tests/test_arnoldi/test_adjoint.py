import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
import pytest_cases
from matfree import test_util
from matfree_extensions import arnoldi, exp_util


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize("krylov_depth", [1, 5, 10])
@pytest_cases.parametrize(
    "reortho", ["none", "full_with_sparsity", "full_without_sparsity"]
)
def test_vjp(nrows, krylov_depth, reortho):
    # todo: see whether a wrap through Hessenberg(T) makes dH Hessenberg
    # todo: verify that the maths-constraints are correct
    #  (more of a maths check, not a code check)
    # todo: see which components simplify for symmetric matrices
    # todo: test that depth=0 and depth>n raise meaningful errors

    # Create a matrix and a direction as a test-case
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(nrows, nrows))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    def fwd(vector, params):
        return arnoldi.forward(
            lambda s, p: p @ s, vector, krylov_depth, params, reortho=reortho
        )

    # Forward pass
    (Q, H, r, c), vjp = jax.vjp(fwd, v, A)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)

    # Call the auto-diff VJP
    dv_ref, dp_ref = vjp((dQ, dH, dr, dc))

    # Call the custom VJP
    (dv, dp), _ = arnoldi.adjoint(
        lambda s, p: p @ s,
        A,
        Q=Q,
        H=H,
        r=r,
        c=c,
        dQ=dQ,
        dH=dH,
        dr=dr,
        dc=dc,
        reortho=reortho,
    )

    # Verify the shapes
    assert dp.shape == (nrows, nrows)
    assert dv.shape == (nrows,)

    # Tie the tolerance to the floating-point accuracy
    small_value = 10 * jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}
    print()
    print(dv)
    print()
    print(dv_ref)
    print()
    print(dp)
    print(dp_ref)

    # Assert gradients match
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(dp, dp_ref, **tols)


@pytest_cases.parametrize("nrows", [15])
@pytest_cases.parametrize("krylov_depth", [10])
@pytest_cases.parametrize("reortho", ["full_with_sparsity", "full_without_sparsity"])
def test_reorthogonalisation(nrows, krylov_depth, reortho):
    # Enable double precision because without, autodiff fails.
    jax.config.update("jax_enable_x64", True)

    # Create a matrix and a direction as a test-case
    A = _lower(exp_util.hilbert(nrows))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    def matvec(s, p):
        # Force sparsity in dp
        # Such an operation has no effect on the numerical values
        # of the forward pass, but tells the vector-Jacobian product
        # which values are irrelevant (i.e. zero). So by using
        # this mask, we get gradients whose sparsity pattern
        # matches that of p. The result da + da.T is not affected.
        p = jnp.tril(p)

        # Evaluate
        return (p + p.T) @ s

    def fwd(vector, params):
        return arnoldi.forward(matvec, vector, krylov_depth, params, reortho=reortho)

    # Forward pass
    (Q, H, r, c), vjp = jax.vjp(fwd, v, A)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)

    # Call the auto-diff VJP
    dv_ref, dp_ref = vjp((dQ, dH, dr, dc))

    # Call the custom VJP
    (dv, dp), _ = arnoldi.adjoint(
        matvec, A, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc, reortho=reortho
    )

    # Verify the shapes
    assert dp.shape == (nrows, nrows)
    assert dv.shape == (nrows,)

    # Tie the tolerance to the floating-point accuracy
    small_value = 10 * jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Assert gradients match
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(dp, dp_ref, **tols)


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize(
    "reortho", ["none", "full_without_sparsity", "full_with_sparsity"]
)
def test_sparsity_in_sigma(nrows, reortho):
    # Create a matrix and a direction as a test-case
    eigvals = 1.1 ** jnp.arange(-nrows // 2, nrows // 2, step=1.0)
    A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    def matvec(s, p):
        return p @ s

    def fwd(vector, params):
        krylov_depth = nrows
        return arnoldi.forward(matvec, vector, krylov_depth, params, reortho=reortho)

    # Forward pass
    (Q, H, r, c), vjp = jax.vjp(fwd, v, A)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)
    # Call the custom VJP
    _, multipliers = arnoldi.adjoint(
        matvec, A, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc, reortho=reortho
    )

    # Tie the tolerance to the floating-point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    Sigma = multipliers["Sigma"]
    assert jnp.allclose(Sigma, jnp.tril(Sigma, -2), **tols)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    # Deterministic values because random seeds
    # would change with double precision
    flat_like = jnp.arange(1.0, 1.0 + len(flat)) / len(flat)

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)


def _lower(m):
    m_tril = jnp.tril(m)
    return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))


@pytest_cases.parametrize("reortho", [True, "full", "None"])
def test_type_error_for_wrong_reorthogonalisation_flag(reortho):
    eye = jnp.eye(1)
    one = jnp.ones((1,))
    kwargs = {
        "Q": eye,
        "H": eye,
        "r": one,
        "c": 1.0,
        "dQ": eye,
        "dH": eye,
        "dr": one,
        "dc": 1.0,
    }
    with pytest.raises(TypeError, match="Unexpected input"):
        arnoldi.adjoint(lambda s: s, **kwargs, reortho=reortho)
