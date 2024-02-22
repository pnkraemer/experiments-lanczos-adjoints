import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def f(x, *, p=2):
    return jnp.sin(p * x**2)


def k(x, y, *, scale_in, scale_out):
    diff = x - y
    log_k = scale_in * jnp.dot(diff, diff)
    return scale_out * jnp.exp(log_k)


def parametrize(fun, **parameters):
    return functools.partial(fun, **parameters)


def vect(fun):
    tmp = jax.vmap(fun, in_axes=(0, None), out_axes=0)
    return jax.vmap(tmp, in_axes=(None, 1), out_axes=1)


def condition(inputs, targets, *, kernel, observation_noise):
    K = kernel(inputs[..., None], inputs[None, ...])
    shift = observation_noise * jnp.eye(len(K))
    return jnp.linalg.solve(K + shift, targets)


# Training data
xs = jnp.linspace(0, 1, num=100, endpoint=True)
ys = f(xs)

# Parametrize and condition
params = {"scale_in": 1.0, "scale_out": 1.0}
k_p = parametrize(k, **params)
k_vect = vect(k_p)
coeff = condition(xs, ys, kernel=k_vect, observation_noise=1.0)

# Evaluate
xs_new = jnp.linspace(0, 1, num=33, endpoint=True)
ys_new = k_vect(xs_new[:, None], xs[None, :]) @ coeff


plt.plot(xs, ys, label="Truth")
plt.plot(xs_new, ys_new, label="Estimate")
plt.xlim((jnp.amin(xs), jnp.amax(xs)))
plt.legend()
plt.show()
