"""Measure the loss of orthogonality."""

import jax
import jax.flatten_util
import jax.numpy as jnp
from matfree_extensions import arnoldi

# from matfree_extensions.util import exp_util

# jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(1)
key_A, key_v = jax.random.split(key, num=2)

for reortho in ["full", "none"]:
    for match in ["match", "full", "none"]:
        print(reortho, match)
        for n in reversed([2, 3, 5, 8, 13, 21, 34]):
            # todo: make this matrix 100x100?
            # A = exp_util.hilbert(n)
            A = jax.random.normal(key_A, shape=(n, n))

            algorithm = arnoldi.hessenberg(
                lambda s, p: p @ s, n, reortho=reortho, reortho_vjp=match
            )
            flat, unflatten = jax.flatten_util.ravel_pytree(A)

            v = jax.random.normal(key_v, shape=(n,))
            Q, *_ = algorithm(v, A)

            @jax.jit
            @jax.jacrev
            def identity(x):
                a = unflatten(x)
                q, h, r, c = algorithm(v, a)
                return jax.flatten_util.ravel_pytree(q @ h @ q.T)[0]

            diff = jnp.eye(len(flat)) - identity(flat)
            error = jnp.sqrt(jnp.mean(diff**2))
            ortho_loss = jnp.linalg.norm(Q.T @ Q - jnp.eye(len(Q.T))) / jnp.sqrt(
                len(Q.T)
            )
            print(A.size, error, ortho_loss)
        print()
    print()
