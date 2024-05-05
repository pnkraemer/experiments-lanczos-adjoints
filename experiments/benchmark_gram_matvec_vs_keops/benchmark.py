import functools
import os
import time

import jax
import jax.numpy as jnp
from matfree_extensions.util import exp_util


def k(x, y, p):
    """Evaluate a square-exponential kernel."""
    diff = x - y
    return p[0] * jnp.exp(-p[1] * jnp.dot(diff, diff))


@functools.partial(jax.jit, static_argnums=[0])
def gram_map_checkpt(f, x, y, p, v):
    """Evaluate a Gram-matrix-vector-product."""

    @jax.checkpoint
    def fx(z):
        fun_v = jax.vmap(f, in_axes=(None, 0, None), out_axes=-1)
        return jax.tree_util.tree_map(lambda s: jnp.dot(s, v), fun_v(z, y, p))

    # Map over rows to reduce memory
    return jax.lax.map(fx, x)


@functools.partial(jax.jit, static_argnums=[0])
def gram_map(f, x, y, p, v):
    """Evaluate a Gram-matrix-vector-product."""

    def fx(z):
        fun_v = jax.vmap(f, in_axes=(None, 0, None), out_axes=-1)
        return jax.tree_util.tree_map(lambda s: jnp.dot(s, v), fun_v(z, y, p))

    # Map over rows to reduce memory
    return jax.lax.map(fx, x)


def gram_map_fwd(f, x, y, p, v):
    """Evaluate a custom forward-pass for the Gram-matrix-vector-product."""
    return gram_map(f, x, y, p, v), {"p": p, "v": v}


def gram_map_bwd(f, x, y, cache, df):
    """Evaluate a custom backward-pass for the Gram-matrix-vector-product."""
    dv, tmp = gram_map(jax.value_and_grad(f, argnums=2), y, x, cache["p"], df)
    return cache["v"].T @ tmp, dv


def gradient(gram_fun, f, x, y):
    """Evaluate the gradient of a vector-Gram-matrix-vector-product."""
    fun = functools.partial(gram_fun, f, x, y)

    def loss(p, v):
        key = jax.random.PRNGKey(1)
        u = jax.random.normal(key, shape=v.shape, dtype=v.dtype)
        return u @ fun(p, v)

    return jax.jit(jax.grad(loss, argnums=(0, 1)))


# Assigning the AD functions
gram_map_ad = gram_map

gram_map_custom = gram_map
gram_map_custom = jax.custom_vjp(gram_map_custom, nondiff_argnums=[0, 1, 2])
gram_map_custom.defvjp(gram_map_fwd, gram_map_bwd)

Ns = 2 ** jnp.arange(5, 17, step=1, dtype=int)
ts_fwd = []
ts_bwd_custom = []
ts_bwd_checkpt = []
ts_bwd_ad = []

for N in Ns:
    print(f"\nN = {N}")
    # Create some test data
    X = jnp.linspace(0, 1, num=N)
    Y = jnp.linspace(0, 1, num=N)
    params = jnp.asarray([1.0, 1.0])
    vector = jnp.linspace(0, 1, num=N)

    print("Benchmark the forward pass.")
    gram_map(k, X, Y, params, vector).block_until_ready()  # pre-compile
    t0 = time.perf_counter()
    gram_map(k, X, Y, params, vector).block_until_ready()
    ts_fwd.append(time.perf_counter() - t0)

    if N <= 17_000:
        print("Benchmark the forward+backward pass (autodiff).")
        # Pre-compile
        (d0, d1) = gradient(gram_map_ad, k, X, Y)(params, vector)
        d0.block_until_ready()
        d1.block_until_ready()

        t0 = time.perf_counter()
        (d0, d1) = gradient(gram_map_ad, k, X, Y)(params, vector)
        d0.block_until_ready()
        d1.block_until_ready()
        ts_bwd_ad.append(time.perf_counter() - t0)

    print("Benchmark the forward+backward pass (autodiff+checkpoint).")
    # Pre-compile
    (d0, d1) = gradient(gram_map_checkpt, k, X, Y)(params, vector)
    d0.block_until_ready()
    d1.block_until_ready()

    t0 = time.perf_counter()
    (d0, d1) = gradient(gram_map_checkpt, k, X, Y)(params, vector)
    d0.block_until_ready()
    d1.block_until_ready()
    ts_bwd_checkpt.append(time.perf_counter() - t0)

    print("Benchmark the forward+backward pass (custom).")
    # Pre-compile
    (d0, d1) = gradient(gram_map_custom, k, X, Y)(params, vector)
    d0.block_until_ready()
    d1.block_until_ready()

    t0 = time.perf_counter()
    (d0, d1) = gradient(gram_map_custom, k, X, Y)(params, vector)
    d0.block_until_ready()
    d1.block_until_ready()
    ts_bwd_custom.append(time.perf_counter() - t0)


print("Saving to a file")
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)


jnp.save(f"{directory}/Ns.npy", jnp.asarray(Ns))
jnp.save(f"{directory}/ts_fwd.npy", jnp.asarray(ts_fwd))
jnp.save(f"{directory}/ts_bwd_checkpt.npy", jnp.asarray(ts_bwd_checkpt))
jnp.save(f"{directory}/ts_bwd_custom.npy", jnp.asarray(ts_bwd_custom))
jnp.save(f"{directory}/ts_bwd_ad.npy", jnp.asarray(ts_bwd_ad))
