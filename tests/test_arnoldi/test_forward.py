import jax
import jax.flatten_util
import jax.numpy as jnp
from matfree_extensions import arnoldi

# Make the prints human-readable
jnp.set_printoptions(2, suppress=True)


def test_decomposition(n=10):
    # todo: use a Hilbert matrix to assert numerical stability

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
    # todo: use a scan instead of a native loop
    # todo: see whether re-orthogonalisation is possible (we know Lambda^\top Q!)
    # todo: see whether a wrap through Hessenberg(T) makes dH Hessenberg
    # todo: verify that the maths-constraints are correct
    #  (more of a maths check, not a code check)
    # todo: use a Hilbert matrix to verify numerical stability
    # todo: see which components simplify for symmetric matrices
    # todo: test corner cases k=0, k=1, k=n

    # A random, asymmetric matrix and a random direction as a test-bed
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(n, n))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(n,))

    # Krylov depth
    k = n // 2

    def fwd(matrix, vector):
        return arnoldi.forward(matrix, vector, k)

    # Forward pass
    (Q, H, r, c), vjp = jax.vjp(fwd, A, v)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)

    # Reference VJP
    da_ref, dv_ref = vjp((dQ, dH, dr, dc))

    # Custom backward pass
    (da, dv), _ = arnoldi.backward(
        A, v, k, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc
    )

    # Tolerance tied to accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Assert gradients match
    assert jnp.allclose(dv, dv_ref, **tols)
    assert jnp.allclose(da, da_ref, **tols)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    key = jax.random.PRNGKey(1)
    flat_like = 0.1 + jax.random.uniform(key, shape=flat.shape, dtype=flat.dtype)

    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)
