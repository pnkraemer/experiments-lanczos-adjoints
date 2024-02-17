import jax
import jax.numpy as jnp
from matfree_extensions import arnoldi

jnp.set_printoptions(2, suppress=True)


def test_decomposition(n=10):
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


def test_vjp(n=10):
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(n, n))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(n,))
    v = v

    k = n // 2

    def fwd(matrix, vector):
        return arnoldi.forward(matrix, vector, k)

    (Q, H, r, c), vjp = jax.vjp(fwd, A, v)

    da_ref, dv_ref = vjp((Q, H, r, c))

    (da, dv), _ = arnoldi.backward(A, v, k, Q=Q, H=H, r=r, c=c, dQ=Q, dH=H, dr=r, dc=c)

    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    print(da)
    print()
    print(da_ref)
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(da, da_ref, **tols)

    assert False
