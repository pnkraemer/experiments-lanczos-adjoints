import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
import pytest_cases
from matfree_extensions import arnoldi, exp_util

jnp.set_printoptions(2, suppress=False)


@pytest_cases.parametrize("nrows", [12])
@pytest_cases.parametrize("krylov_depth", [12])
def test_vjp(nrows, krylov_depth):
    # todo: see whether a wrap through Hessenberg(T) makes dH Hessenberg
    # todo: verify that the maths-constraints are correct
    #  (more of a maths check, not a code check)
    # todo: see which components simplify for symmetric matrices
    # todo: test that depth=0 and depth>n raise meaningful errors

    # Create a random, asymmetric matrix and a random direction as a test-bed
    A = exp_util.hilbert(nrows)
    v = jnp.ones((nrows,))

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
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Assert gradients match
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(da, da_ref, **tols)

    pytest.fail(".")


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    # Random values (strictly positive and bounded away from zero for good measure)
    key = jax.random.PRNGKey(1)
    flat_like = 0.1 + 0.9 * jax.random.uniform(key, shape=flat.shape, dtype=flat.dtype)
    flat_like = jnp.arange(1.0, 1.0 + len(flat)) / len(flat)

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
