import functools
import time

import jax
import jax.experimental.sparse
import jax.numpy as jnp
from matfree import hutchinson, slq
from matfree_extensions import bcoo_random_spd, integrand_slq_spd_value_and_grad

n = 1_000  # nrows/ncols
nz = 2 * n // 10  # nonzeros
seed = 1
num_samples = 10
num_reps = 1

key = jax.random.PRNGKey(seed)
key_mat, key_estim1, key_estim2 = jax.random.split(key, num=3)

x_like = jnp.ones((n,))
sampler = hutchinson.sampler_normal(x_like, num=num_samples)


def matvec(x, p):
    return p @ x


M = bcoo_random_spd(key_mat, num_rows=n, num_nonzeros=nz)


@functools.partial(jax.experimental.sparse.value_and_grad, allow_int=True)
def logdet(p):
    M = p.todense()
    return jnp.linalg.slogdet(M)[1]


value_true, grad_true = logdet(M)
print(value_true)


def error(g):
    return jnp.linalg.norm((g - grad_true.data) / (1 + grad_true.data)) / jnp.sqrt(
        grad_true.data.size
    )


for order in 2.0 ** (jnp.arange(10)):
    order = int(order)
    # Approximation
    integrand = integrand_slq_spd_value_and_grad(
        jnp.log, order, matvec, grad_func=jax.experimental.sparse.grad
    )
    estimate_approximate = hutchinson.hutchinson(integrand, sampler)
    estimate_approximate = jax.jit(estimate_approximate)
    fx, grad = estimate_approximate(key_estim1, M)
    t0 = time.perf_counter()
    for _ in range(num_reps):
        _, grad = estimate_approximate(key_estim1, M)
        grad.block_until_ready()
    t1 = time.perf_counter()
    print(order, error(grad), (t1 - t0) / num_reps)

print()
for order in 2.0 ** (jnp.arange(7)):
    order = int(order)

    # Reference
    integrand = slq.integrand_slq_spd(jnp.log, order, matvec)
    estimate_reference = hutchinson.hutchinson(integrand, sampler)
    estimate_reference = jax.jit(
        jax.experimental.sparse.value_and_grad(
            estimate_reference, allow_int=True, argnums=1
        )
    )
    _fx, grad = estimate_reference(key_estim1, M)
    t0 = time.perf_counter()
    for _ in range(num_reps):
        _, grad = estimate_reference(key_estim1, M)
        grad.block_until_ready()
    t1 = time.perf_counter()
    print(order, error(grad.data), (t1 - t0) / num_reps)
