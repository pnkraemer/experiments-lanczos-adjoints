import jax.numpy as jnp


def forward(A, v, krylov_depth):
    (n,) = jnp.shape(v)
    k = krylov_depth

    H = jnp.zeros((k, k))
    Q = jnp.zeros((n, k))
    r = jnp.zeros((n,))

    return Q, H, r
