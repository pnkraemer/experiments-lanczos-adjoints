import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
import pytest_cases
from matfree_extensions import arnoldi, exp_util


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize("krylov_depth", [1, 5, 10])
def test_decomposition_with_reortho(nrows, krylov_depth):
    # Make the prints human-readable
    jnp.set_printoptions(2, suppress=True)

    # Create an ill-conditioned test-matrix (that requires reortho=True)
    A = exp_util.hilbert(nrows)
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    # Decompose
    Q, H, r, c = arnoldi.forward(A, v, krylov_depth, reortho=True)

    # Assert shapes
    assert Q.shape == (nrows, krylov_depth)
    assert H.shape == (krylov_depth, krylov_depth)
    assert r.shape == (nrows,)
    assert c.shape == ()

    # Tie the test-strictness to the floating point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Test the decompositions
    e0, ek = jnp.eye(krylov_depth)[[0, -1], :]
    assert jnp.allclose(A @ Q - Q @ H - jnp.outer(r, ek), 0.0, **tols)
    assert jnp.allclose(Q.T @ Q - jnp.eye(krylov_depth), 0.0, **tols)
    assert jnp.allclose(Q @ e0, c * v, **tols)


def test_decomposition_raises_error_for_wrong_depth_too_small():
    with pytest.raises(ValueError, match="depth"):
        _ = arnoldi.forward(jnp.eye(2), jnp.ones((2,)), 0)


def test_decomposition_raises_error_for_wrong_depth_too_high():
    with pytest.raises(ValueError, match="depth"):
        _ = arnoldi.forward(jnp.eye(2), jnp.ones((2,)), 3)
