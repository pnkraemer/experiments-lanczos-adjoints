import functools
import os
import time

import jax
import jax.numpy as jnp
from matfree_extensions.util import exp_util

jax.config.update("jax_enable_compilation_cache", False)

MAX_BATCH = 2**30
"""We can fit MAX_BATCH * 32 bit values in memory."""


def k(x, y, p):
    """Evaluate a square-exponential kernel."""
    # maaaan, this is factor ~10 times faster than dot(x-y, x-y)
    norm_sq = jnp.dot(x, x) + jnp.dot(y, y) - 2 * jnp.dot(x, y)

    return p[0] * jnp.exp(-p[1] * norm_sq)


@functools.partial(jax.jit, static_argnums=[0, 5])
def gram_map_checkpt(f, x, y, p, v, batch_size):
    """Evaluate a Gram-matrix-vector-product."""

    def fx(z):
        fun_v = jax.vmap(f, in_axes=(None, 0, None), out_axes=-1)
        return jax.tree_util.tree_map(lambda s: jnp.dot(s, v), fun_v(z, y, p))

    n, *shape = jnp.shape(x)
    x_ = jnp.reshape(x, (n // batch_size, batch_size, *shape))
    mapped = jax.lax.map(jax.checkpoint(jax.vmap(fx)), x_)
    return jax.tree_util.tree_map(lambda s: s.reshape((-1,)), mapped)


@functools.partial(jax.jit, static_argnums=[0, 5])
def gram_map(f, x, y, p, v, batch_size):
    """Evaluate a Gram-matrix-vector-product."""

    def fx(z):
        fun_v = jax.vmap(f, in_axes=(None, 0, None), out_axes=-1)
        return jax.tree_util.tree_map(lambda s: jnp.dot(s, v), fun_v(z, y, p))

    n, *shape = jnp.shape(x)
    x_ = jnp.reshape(x, (n // batch_size, batch_size, *shape))
    mapped = jax.lax.map(jax.vmap(fx), x_)
    return jax.tree_util.tree_map(
        lambda s: s.reshape((s.shape[0] * s.shape[1], -1)).squeeze(), mapped
    )


def gram_map_fwd(f, x, y, p, v, batch_size):
    """Evaluate a custom forward-pass for the Gram-matrix-vector-product."""
    return gram_map(f, x, y, p, v, batch_size), {"p": p, "v": v}


def gram_map_bwd(f, x, y, batch_size, cache, df):
    """Evaluate a custom backward-pass for the Gram-matrix-vector-product."""
    # For some reason, two calls (one with f, one with jacrev)
    # beat one call (value_and_grad).
    # dv = gram_map(f, y, x, cache["p"], df, batch_size)
    # tmp = gram_map(jax.jacrev(f, argnums=2), y, x, cache["p"], df, batch_size)
    dv, tmp = gram_map(
        jax.value_and_grad(f, argnums=2), y, x, cache["p"], df, batch_size
    )
    return cache["v"].T @ tmp, dv


def gradient(gram_fun, f, x, y, batch_size):
    """Evaluate the gradient of a vector-Gram-matrix-vector-product."""

    def loss(p, v):
        return gram_fun(f, x, y, p, v, batch_size)

    def full(p, v):
        fwd, vjp = jax.vjp(loss, p, v)
        return fwd, vjp(fwd)

    return jax.jit(full)


# Assigning the AD functions
gram_map_ad = gram_map

gram_map_custom = gram_map
gram_map_custom = jax.custom_vjp(gram_map_custom, nondiff_argnums=[0, 1, 2, 5])
gram_map_custom.defvjp(gram_map_fwd, gram_map_bwd)

Ns = 2 ** jnp.arange(8, 14, step=1, dtype=int)
ts_fwd = []
ts_bwd_custom = []
ts_bwd_checkpt = []
ts_bwd_ad = []

dim = 16

for N in Ns:
    batch_size = jnp.minimum(N, MAX_BATCH // N)
    print(f"\nN = {N}, batch_size = {batch_size}")

    # Create some test data
    X = jnp.linspace(0, 1, num=N)
    X = jnp.stack([X] * dim, axis=1)
    Y = X
    # Y = jnp.linspace(0, 1, num=N)
    params = jnp.asarray([1.0, 1.0])
    vector = jnp.linspace(0, 1, num=N)

    print("Benchmark the forward pass.", end=" ")
    fun = jax.jit(gram_map, static_argnums=[0, 5])
    fun(k, X, Y, params, vector, int(batch_size)).block_until_ready()  # pre-compile
    t0 = time.perf_counter()
    fun(k, X, Y, params, vector, int(batch_size)).block_until_ready()
    t1 = time.perf_counter() - t0
    ts_fwd.append(t1)
    print(t1)

    if N <= 16_000:
        print("Benchmark the forward+backward pass (autodiff).", end=" ")
        # Pre-compile
        fun = jax.jit(gradient(gram_map_ad, k, X, Y, int(batch_size)))
        v0, (d0, d1) = fun(params, vector)
        v0.block_until_ready()
        d0.block_until_ready()
        d1.block_until_ready()

        t0 = time.perf_counter()
        v0, (d0, d1) = fun(params, vector)
        v0.block_until_ready()
        d0.block_until_ready()
        d1.block_until_ready()
        t1 = time.perf_counter() - t0
        ts_bwd_ad.append(t1)
        print(t1)

    print("Benchmark the forward+backward pass (autodiff+checkpoint).", end=" ")
    # Pre-compile
    fun = jax.jit(gradient(gram_map_checkpt, k, X, Y, int(batch_size)))

    v0, (d0, d1) = fun(params, vector)
    v0.block_until_ready()
    d0.block_until_ready()
    d1.block_until_ready()

    t0 = time.perf_counter()
    v0, (d0, d1) = fun(params, vector)
    v0.block_until_ready()
    d0.block_until_ready()
    d1.block_until_ready()
    t1 = time.perf_counter() - t0
    ts_bwd_checkpt.append(t1)
    print(t1)

    print("Benchmark the forward+backward pass (custom).", end=" ")
    # Pre-compile
    fun = jax.jit(gradient(gram_map_custom, k, X, Y, batch_size=int(batch_size)))

    v0, (d0, d1) = fun(params, vector)
    v0.block_until_ready()
    d0.block_until_ready()
    d1.block_until_ready()

    t0 = time.perf_counter()
    v0, (d0, d1) = fun(params, vector)
    v0.block_until_ready()
    d0.block_until_ready()
    d1.block_until_ready()
    t1 = time.perf_counter() - t0
    ts_bwd_custom.append(t1)
    print(t1)


print("Saving to a file")
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)


jnp.save(f"{directory}/Ns.npy", jnp.asarray(Ns))
jnp.save(f"{directory}/ts_fwd.npy", jnp.asarray(ts_fwd))
jnp.save(f"{directory}/ts_bwd_checkpt.npy", jnp.asarray(ts_bwd_checkpt))
jnp.save(f"{directory}/ts_bwd_custom.npy", jnp.asarray(ts_bwd_custom))
jnp.save(f"{directory}/ts_bwd_ad.npy", jnp.asarray(ts_bwd_ad))
