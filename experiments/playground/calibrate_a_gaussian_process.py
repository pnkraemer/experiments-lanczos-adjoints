import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def f(x):
    return jnp.sin(10 * x**2)


def k(x, y, scale_in, scale_out):
    diff = x - y
    log_k = scale_in * jnp.dot(diff, diff)
    return scale_out * jnp.exp(log_k)


xs = jnp.linspace(0, 1, num=100, endpoint=True)
ys = f(xs)


def condition(inputs, targets, scale_in, scale_out, observation_noise):
    k_p = functools.partial(k, scale_in=scale_in, scale_out=scale_out)
    k_p_1 = jax.vmap(k_p, in_axes=(0, None), out_axes=0)
    k_p_2 = jax.vmap(k_p_1, in_axes=(None, 1), out_axes=1)

    K = k_p_2(inputs[..., None], inputs[None, ...])
    shift = observation_noise * jnp.eye(len(K))
    return jnp.linalg.solve(K + shift, targets)


ms = condition(xs, ys, 1, 1, 1.0)


plt.plot(xs, ys)
plt.plot(xs, ms)
plt.show()
