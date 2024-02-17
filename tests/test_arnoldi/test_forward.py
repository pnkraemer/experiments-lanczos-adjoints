import jax
import jax.flatten_util
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
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)
    da_ref, dv_ref = vjp((dQ, dH, dr, dc))

    (da, dv), _ = arnoldi.backward(
        A, v, k, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc
    )

    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    print(da)
    print()
    print(da_ref)
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(da, da_ref, **tols)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    key = jax.random.PRNGKey(1)
    flat_like = 0.1 + jax.random.uniform(key, shape=flat.shape, dtype=flat.dtype)
    # flat_like = flat_like * flat
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
