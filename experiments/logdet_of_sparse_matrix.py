import functools

import jax
import jax.experimental.sparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import hutchinson, slq

from matfree_extensions import (bcoo_random_spd, integrand_slq_spd_custom_vjp,
                                integrand_slq_spd_value_and_grad)

n = 100  # nrows/ncols
nz = 20  # nonzeros
seed = 1
num_samples = 1000

key = jax.random.PRNGKey(seed)
key_mat, key_estim1, key_estim2 = jax.random.split(key, num=3)

x_like = jnp.ones((n,))
sampler = hutchinson.sampler_normal(x_like, num=num_samples)


def matvec(x, p):
    return p.T @ (p @ x)


M = bcoo_random_spd(key_mat, num_rows=n, num_nonzeros=nz)


@functools.partial(jax.experimental.sparse.value_and_grad, allow_int=True)
def logdet(p):
    M = p.todense().T @ p.todense()
    return jnp.linalg.slogdet(M)[1]


grad_true = logdet(M)[1]


def error(g):
    return jnp.linalg.norm((g - grad_true.data))


for order in 2.0 ** (jnp.arange(7)):
    order = int(order)
    # Approximation
    integrand = integrand_slq_spd_value_and_grad(
        jnp.log, order, matvec, grad_func=jax.experimental.sparse.grad
    )
    estimate_approximate = hutchinson.hutchinson(integrand, sampler)
    estimate_approximate = jax.jit(estimate_approximate)
    fx, grad = estimate_approximate(key_estim1, M)
    print(order, error(grad))

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
    print(order, error(grad.data))

#
# vmin, vmax = -10, 10
#
# mosaic = [["truth", "backprop-slq", "backprop-naive"],
#                               ["error-truth", "error-backprop-slq", "error-backprop-naive"]]
# fig, ax = plt.subplot_mosaic(mosaic, sharex=True, sharey=True)
#
# ax["truth"].set_ylabel("Gradient")
# ax["error-truth"].set_ylabel("Error (gradient)")
#
#
# ax["truth"].spy(grad_true.todense())
# ax["error-truth"].imshow(jnp.log10(1e-10 + jnp.abs((grad_true-grad_true).todense())), vmin=vmin, vmax=vmax)
#
# ax["backprop-slq"].spy(grad.todense())
# ax["error-backprop-slq"].imshow(jnp.log10(1e-10 + jnp.abs((grad-grad_true).todense())), vmin=vmin, vmax=vmax)
#
#
# ax["backprop-naive"].spy(grad.todense())
# img = ax["error-backprop-naive"].imshow(jnp.log10(1e-10 + jnp.abs((grad-grad_true).todense())), vmin=vmin, vmax=vmax)
#
# plt.colorbar(img)
# plt.show()
