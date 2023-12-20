import time

import jax
import jax.experimental.sparse
import jax.numpy as jnp
from matfree import hutchinson, slq

from matfree_extensions import integrand_slq_spd_custom_vjp

n = 100  # nrows/ncols
seed = 1
num_samples = 100
num_reps = 1

key = jax.random.PRNGKey(seed)
key_mat, key_estim1, key_estim2 = jax.random.split(key, num=3)

x_like = jnp.ones((n,))
sampler = hutchinson.sampler_rademacher(x_like, num=num_samples)


def matvec(x, p):
    return p * x


@jax.value_and_grad
def logdet(p):
    return jnp.sum(jnp.log(p))
    # M = jax.jacfwd(lambda x: matvec(x, p))(x_like)
    # return jnp.linalg.slogdet(M)[1]


parameters = 1 + jax.random.uniform(key_estim2, shape=(n,))
# parameters /= jnp.linalg.norm(parameters)
value_true, grad_true = logdet(parameters)

print(value_true)


def error(g):
    return jnp.linalg.norm((g - grad_true) / (1 + grad_true)) / jnp.sqrt(grad_true.size)


for order in 2.0 ** (jnp.arange(5)):
    order = int(order)
    # Approximation
    integrand = integrand_slq_spd_custom_vjp(jnp.log, order, matvec)
    integrand = jax.value_and_grad(integrand, allow_int=True, argnums=1)
    estimate_approximate = hutchinson.hutchinson(integrand, sampler)
    estimate_approximate = jax.jit(estimate_approximate)
    fx, grad = estimate_approximate(key_estim1, parameters)
    t0 = time.perf_counter()
    for _ in range(num_reps):
        _, grad = estimate_approximate(key_estim1, parameters)
        grad.block_until_ready()
    t1 = time.perf_counter()
    print(order, jnp.abs(fx - value_true), error(grad), (t1 - t0) / num_reps)

print()
for order in 2.0 ** (jnp.arange(5)):
    order = int(order)

    # Reference
    integrand = slq.integrand_slq_spd(jnp.log, order, matvec)
    integrand = jax.value_and_grad(integrand, allow_int=True, argnums=1)
    estimate_reference = hutchinson.hutchinson(integrand, sampler)
    estimate_reference = jax.jit(estimate_reference)
    fx, grad = estimate_reference(key_estim1, parameters)
    t0 = time.perf_counter()
    for _ in range(num_reps):
        _, grad = estimate_reference(key_estim1, parameters)
        grad.block_until_ready()
    t1 = time.perf_counter()
    print(order, jnp.abs(fx - value_true), error(grad), (t1 - t0) / num_reps)
