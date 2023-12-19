import functools

import jax
import jax.experimental.sparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import hutchinson, slq

from matfree_extensions import bcoo_random_spd, integrand_slq_spd_custom_vjp

n = 10  # nrows/ncols
nz = 20  # nonzeros
seed = 1
order = 9

key = jax.random.PRNGKey(seed)
key_mat, key_estim1, key_estim2 = jax.random.split(key, num=3)

x_like = jnp.ones((n,))
sampler = hutchinson.sampler_normal(x_like, num=100)


def matvec(x, p):
    return p.T @ (p @ x)


M = bcoo_random_spd(key_mat, num_rows=n, num_nonzeros=nz)


@functools.partial(jax.experimental.sparse.value_and_grad, allow_int=True)
def logdet(p):
    M = p.todense().T @ p.todense()
    return jnp.linalg.slogdet(M)[1]


print(logdet(M)[1].todense())


# Approximation
integrand = integrand_slq_spd_custom_vjp(jnp.log, order, matvec)
estimate_approximate = hutchinson.hutchinson(integrand, sampler)
estimate_approximate = jax.jit(
    jax.experimental.sparse.value_and_grad(
        estimate_approximate, allow_int=True, argnums=1
    )
)
fx, grad = estimate_approximate(key_estim1, M)
print(fx, grad.todense())


# Reference
integrand = slq.integrand_slq_spd(jnp.log, order, matvec)
estimate_reference = hutchinson.hutchinson(integrand, sampler)
estimate_reference = jax.jit(
    jax.experimental.sparse.value_and_grad(
        estimate_reference, allow_int=True, argnums=1
    )
)
print(estimate_reference(key_estim1, M)[1].todense())
