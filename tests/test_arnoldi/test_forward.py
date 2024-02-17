import jax
import jax.numpy as jnp
from matfree_extensions import arnoldi


def test_sth(n=10):
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(n, n))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(n,))

    k = n // 2
    Q, H, r = arnoldi.forward(A, v, k)

    assert Q.shape == (n, k)
    assert H.shape == (k, k)

    w = jnp.eye(k)[-1]
    assert jnp.allclose(A @ Q - Q @ H - jnp.outer(r, w), 0.0)
    assert jnp.allclose(Q.T @ Q - jnp.eye(k), 0.0)
    assert False
