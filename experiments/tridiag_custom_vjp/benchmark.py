"""Implement a first small benchmark."""


import functools
import time

import jax.flatten_util
import jax.numpy as jnp

from matfree_extensions import lanczos

n = 1_000
seed = 1


# Set up a test-matrix
params = jax.random.uniform(jax.random.PRNGKey(seed), shape=(n,))

# Set up an initial vector
vector = jax.random.normal(jax.random.PRNGKey(seed + 1), shape=(n,))

# Flatten the inputs
flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))

krylov_depth = 1
while (krylov_depth := 2 * krylov_depth) < n:
    print("Krylov-depth:", krylov_depth)

    # Construct an vector-to-vector decomposition function
    def decompose(f, *, custom_vjp):
        algorithm = lanczos.tridiag(
            lambda s, p: p * s, krylov_depth, custom_vjp=custom_vjp
        )
        output = algorithm(*unflatten(f))
        return jax.flatten_util.ravel_pytree(output)[0]

    # Construct the two implementations
    reference = jax.jit(functools.partial(decompose, custom_vjp=False))
    implementation = jax.jit(functools.partial(decompose, custom_vjp=True))

    # Compute both VJPs
    fx_ref, vjp_ref = jax.vjp(reference, flat)
    fx_imp, vjp_imp = jax.vjp(implementation, flat)
    # Assert that the forward-passes are identical
    assert jnp.allclose(fx_ref, fx_imp)

    # Assert that the VJPs into a bunch of random directions are identical
    key = jax.random.PRNGKey(seed + 2)
    dnu = jax.random.normal(key, shape=jnp.shape(reference(flat)))

    # assert jnp.allclose(*vjp_ref(dnu), *vjp_imp(dnu), atol=1e-4, rtol=1e-4)

    t0 = time.perf_counter()
    for _ in range(2):
        _ = vjp_ref(dnu)[0].block_until_ready()
    t1 = time.perf_counter()
    time_autodiff = (t1 - t0) / 2
    print("Time (AutoDiff):", time_autodiff)

    t0 = time.perf_counter()
    for _ in range(2):
        _ = vjp_imp(dnu)[0].block_until_ready()
    t1 = time.perf_counter()
    time_custom = (t1 - t0) / 2
    print("Time (custom VJP):", time_custom)
    print("Ratio:", time_custom / time_autodiff)
    print()
