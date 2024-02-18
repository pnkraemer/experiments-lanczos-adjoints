import jax
import jax.flatten_util
import jax.numpy as jnp
from matfree_extensions import arnoldi

# Make the prints human-readable
jnp.set_printoptions(2, suppress=True)


def test_decomposition(n=10):
    # todo: use a Hilbert matrix to assert numerical stability
    # todo: test corner cases k=0, k=1, k=n as well.

    A = jax.random.normal(jax.random.PRNGKey(1), shape=(n, n))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(n,))
    v = v

    k = n // 2
    Q, H, r, c = arnoldi.forward(A, v, k)

    assert Q.shape == (n, k)
    assert H.shape == (k, k)

    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    w = jnp.eye(k)[-1]
    assert jnp.allclose(A @ Q - Q @ H - jnp.outer(r, w), 0.0, **tols)
    assert jnp.allclose(Q.T @ Q - jnp.eye(k), 0.0, **tols)
    assert jnp.allclose(Q[:, 0], c * v, **tols)
