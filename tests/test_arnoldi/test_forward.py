import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
import pytest_cases
from matfree_extensions import arnoldi


@pytest_cases.parametrize("n", [10])
@pytest_cases.parametrize("k", [1, 5, 9, 10])
def test_decomposition(n, k):
    # todo: use a Hilbert matrix to assert numerical stability
    # todo: test corner cases k=0, k=1, k=n as well.
    # todo: use a scan for the forward pass

    # Make the prints human-readable
    jnp.set_printoptions(2, suppress=True)

    A = jax.random.normal(jax.random.PRNGKey(1), shape=(n, n))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(n,))
    v = v

    Q, H, r, c = arnoldi.forward(A, v, k)

    assert Q.shape == (n, k)
    assert H.shape == (k, k)

    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    e0, ek = jnp.eye(k)[[0, -1], :]
    assert jnp.allclose(A @ Q - Q @ H - jnp.outer(r, ek), 0.0, **tols)
    assert jnp.allclose(Q.T @ Q - jnp.eye(k), 0.0, **tols)
    assert jnp.allclose(Q @ e0, c * v, **tols)


def test_decomposition_raises_error_for_wrong_depth_too_small():
    with pytest.raises(ValueError, match="depth"):
        _ = arnoldi.forward(jnp.eye(2), jnp.ones((2,)), 0)


def test_decomposition_raises_error_for_wrong_depth_too_high():
    with pytest.raises(ValueError, match="depth"):
        _ = arnoldi.forward(jnp.eye(2), jnp.ones((2,)), 3)
