import argparse
import functools
import time

import jax
import jax.numpy as jnp


def k(x, y, p):
    return p[0] * jnp.exp(-p[1] * (x - y) ** 2)


@functools.partial(jax.jit, static_argnums=[0])
def gram(f, x, y, p, v):
    def fx(z):
        fun_v = jax.vmap(f, in_axes=(None, -1, None), out_axes=-1)
        return jnp.dot(fun_v(z, y, p), v)

    return jax.lax.map(fx, x)


def gram_fwd(f, x, y, p, v):
    return gram(f, x, y, p, v), {"p": p, "v": v}


def gram_bwd(f, x, y, cache, df):
    dv = gram(f, y, x, cache["p"], df)
    dp = gram(jax.grad(f, argnums=2), y, x, cache["p"], df)
    return cache["v"].T @ dp, dv


def autodiff(gram_fun, f, x, y):
    fun = functools.partial(gram_fun, f, x, y)

    def loss(p, v):
        key = jax.random.PRNGKey(1)
        u = jax.random.normal(key, shape=v.shape, dtype=v.dtype)
        return u @ fun(p, v)

    return jax.jit(jax.grad(loss, argnums=(0, 1)))


parser = argparse.ArgumentParser()
parser.add_argument("--custom_ad", action=argparse.BooleanOptionalAction)
parser.add_argument("--data_size", type=int, required=True)
args = parser.parse_args()


xs = jnp.linspace(0, 1, num=args.data_size)
ys = xs
ps = jnp.asarray([1.0, 1.0])
vs = xs


if args.custom_ad:
    print("Setting a clever gradient...")
    gram = jax.custom_vjp(gram, nondiff_argnums=[0, 1, 2])
    gram.defvjp(gram_fwd, gram_bwd)

gram(k, xs, ys, ps, vs).block_until_ready()
t0 = time.perf_counter()
gram(k, xs, ys, ps, vs).block_until_ready()
print("Forward pass:", time.perf_counter() - t0)


d0, d1 = autodiff(gram, k, xs, ys)(ps, vs)
d0.block_until_ready()
d1.block_until_ready()

t0 = time.perf_counter()
d0, d1 = autodiff(gram, k, xs, ys)(ps, vs)
d0.block_until_ready()
d1.block_until_ready()
print("Forward+backward pass:", time.perf_counter() - t0)
