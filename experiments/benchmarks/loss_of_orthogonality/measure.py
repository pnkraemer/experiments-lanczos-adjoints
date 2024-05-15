"""Measure the loss of orthogonality."""

import jax
import jax.flatten_util
from matfree_extensions import arnoldi

key = jax.random.PRNGKey(1)
key_A, key_v = jax.random.split(key, num=2)

n = 100
A = jax.random.normal(key_A, shape=(n, n))
v = jax.random.normal(key_v, shape=(n,))

algorithm = arnoldi.hessenberg(lambda s, p: p @ s, n, reortho="full")
flat, unflatten = jax.flatten_util.ravel_pytree(A)


def identity(x):
    a = unflatten(x)
    q, h, r, c = algorithm(v, a)
    return jax.flatten_util.ravel_pytree(q @ h @ q.T)[0]


print(flat)
print(identity(flat))
