import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest_cases
from matfree_extensions import arnoldi, exp_util


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize("krylov_depth", [1, 5, 10])
@pytest_cases.parametrize("reortho", [False, True])
def test_vjp(nrows, krylov_depth, reortho):
    # todo: see whether a wrap through Hessenberg(T) makes dH Hessenberg
    # todo: verify that the maths-constraints are correct
    #  (more of a maths check, not a code check)
    # todo: see which components simplify for symmetric matrices
    # todo: test that depth=0 and depth>n raise meaningful errors

    # Create a matrix and a direction as a test-case
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(nrows, nrows))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    def fwd(matrix, vector):
        return arnoldi.forward(matrix, vector, krylov_depth, reortho=reortho)

    # Forward pass
    (Q, H, r, c), vjp = jax.vjp(fwd, A, v)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)

    # Call the auto-diff VJP
    da_ref, dv_ref = vjp((dQ, dH, dr, dc))

    # Call the custom VJP
    (da, dv) = arnoldi.vjp(
        A, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc, reortho=reortho
    )

    # Verify the shapes
    assert da.shape == (nrows, nrows)
    assert dv.shape == (nrows,)

    # Tie the tolerance to the floating-point accuracy
    small_value = 10 * jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Assert gradients match
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(da, da_ref, **tols)


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize("krylov_depth", [1, 5, 10])
def test_vjp_with_reorthogonalisation(nrows, krylov_depth):
    # Enable double precision because without, autodiff fails.
    jax.config.update("jax_enable_x64", True)

    # Create a matrix and a direction as a test-case
    A = exp_util.hilbert(nrows)
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    def fwd(matrix, vector):
        return arnoldi.forward(matrix, vector, krylov_depth, reortho=True)

    # Forward pass
    (Q, H, r, c), vjp = jax.vjp(fwd, A, v)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)

    # Call the auto-diff VJP
    da_ref, dv_ref = vjp((dQ, dH, dr, dc))

    # Call the custom VJP
    (da, dv) = arnoldi.vjp(
        A, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc, reortho=True
    )

    # Verify the shapes
    assert da.shape == (nrows, nrows)
    assert dv.shape == (nrows,)

    # Tie the tolerance to the floating-point accuracy
    small_value = 10 * jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Assert gradients match
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(da, da_ref, **tols)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    # Deterministic values because random seeds
    # would change with double precision
    flat_like = jnp.arange(1.0, 1.0 + len(flat)) / len(flat)

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
