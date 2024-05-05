import functools

import jax
import jax.numpy as jnp


def k(x, y, p):
    return p[0] * jnp.exp(-p[1] * (x - y) ** 2)


@functools.partial(jax.jit, static_argnums=[0])
def gram(f, x, y, p, v):
    def fx(z):
        return f(z[:, None], y[None, :], p) @ v

    matvec = jax.lax.map(fx, x[:, None])
    return jnp.squeeze(matvec, axis=-1)


N = 100_000
xs = jnp.linspace(0, 1, num=N)
ys = xs
ps = jnp.asarray([1.0, 1.0])
vs = xs

print(gram(k, xs, ys, ps, vs))
