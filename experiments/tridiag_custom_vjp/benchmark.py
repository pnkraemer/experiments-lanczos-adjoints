"""Implement a first small benchmark."""


import functools
import time

import jax.flatten_util
import jax.numpy as jnp

from matfree_extensions import lanczos

n = 10_000
seed = 1


# Set up a test-matrix
params = jax.random.uniform(jax.random.PRNGKey(seed), shape=(n,))

# Set up an initial vector
vector = jax.random.normal(jax.random.PRNGKey(seed + 1), shape=(n,))

# Flatten the inputs
flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))

krylov_depth = 1
while (krylov_depth := 2 * (krylov_depth)) < n:
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

    # Compute a VJP into a random direction
    # (This ignores potential symmetry/orthogonality constraints of the outputs.
    # But we only care about speed at this point, so it is fine.)
    key = jax.random.PRNGKey(seed + 2)
    dnu = jax.random.normal(key, shape=jnp.shape(reference(flat)))

    fx_imp, vjp_imp = jax.vjp(implementation, flat)
    vjp_imp = jax.jit(vjp_imp)
    _ = vjp_imp(dnu)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(2):
        _ = vjp_imp(dnu)[0].block_until_ready()
    t1 = time.perf_counter()
    time_custom = (t1 - t0) / 2
    print("Time (custom VJP):\n\t", time_custom)

    if krylov_depth < 150:
        fx_ref, vjp_ref = jax.vjp(reference, flat)
        vjp_ref = jax.jit(vjp_ref)

        _ = vjp_ref(dnu)[0].block_until_ready()

        t0 = time.perf_counter()
        for _ in range(2):
            _ = vjp_ref(dnu)[0].block_until_ready()
        t1 = time.perf_counter()
        time_autodiff = (t1 - t0) / 2
        print("Time (AutoDiff):\n\t", time_autodiff)

        diff = vjp_ref(dnu)[0] - vjp_imp(dnu)[0]
        diff = jnp.linalg.norm(diff / jnp.abs(vjp_imp(dnu)[0])) / jnp.sqrt(diff.size)
        print("Norm of output difference:\n\t", diff)
        print("Ratio (small is good):\n\t", time_custom / time_autodiff)

    print()
