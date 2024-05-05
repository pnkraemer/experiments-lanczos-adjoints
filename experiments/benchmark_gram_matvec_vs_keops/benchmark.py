import argparse
import functools
import time

import jax
import jax.numpy as jnp


def k(x, y, p):
    """Evaluate a square-exponential kernel."""
    diff = x - y
    return p[0] * jnp.exp(-p[1] * jnp.dot(diff, diff))


@functools.partial(jax.jit, static_argnums=[0])
def gram(f, x, y, p, v):
    """Evaluate a Gram-matrix-vector-product."""

    def fx(z):
        fun_v = jax.vmap(f, in_axes=(None, 0, None), out_axes=-1)
        return jnp.dot(fun_v(z, y, p), v)

    # Map over rows to reduce memory
    return jax.lax.map(fx, x)


def autodiff(gram_fun, f, x, y):
    """Evaluate the gradient of a vector-Gram-matrix-vector-product."""
    fun = functools.partial(gram_fun, f, x, y)

    def loss(p, v):
        key = jax.random.PRNGKey(1)
        u = jax.random.normal(key, shape=v.shape, dtype=v.dtype)
        return u @ fun(p, v)

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))


def gram_fwd(f, x, y, p, v):
    """Evaluate a custom forward-pass for the Gram-matrix-vector-product."""
    return gram(f, x, y, p, v), {"p": p, "v": v}


def gram_bwd(f, x, y, cache, df):
    """Evaluate a custom backward-pass for the Gram-matrix-vector-product."""
    dv = gram(f, y, x, cache["p"], df)
    dp = gram(jax.grad(f, argnums=2), y, x, cache["p"], df)
    return cache["v"].T @ dp, dv


parser = argparse.ArgumentParser()
parser.add_argument("--custom_ad", action=argparse.BooleanOptionalAction)
parser.add_argument("--data_size", type=int, required=True)
args = parser.parse_args()

# Create some test data
X = jnp.linspace(0, 1, num=args.data_size)
Y = jnp.linspace(0, 1, num=args.data_size)
params = jnp.asarray([1.0, 1.0])
vector = jnp.linspace(0, 1, num=args.data_size)


if args.custom_ad:
    print("Setting a clever gradient...")
    gram = jax.custom_vjp(gram, nondiff_argnums=[0, 1, 2])
    gram.defvjp(gram_fwd, gram_bwd)

print("Benchmark the forward pass.")
gram(k, X, Y, params, vector).block_until_ready()  # pre-compile
t0 = time.perf_counter()
gram(k, X, Y, params, vector).block_until_ready()
print("\tRun time:", time.perf_counter() - t0)


print("Benchmark the forward+backward pass.")

# Pre-compile
val, (d0, d1) = autodiff(gram, k, X, Y)(params, vector)
val.block_until_ready()
d0.block_until_ready()
d1.block_until_ready()

t0 = time.perf_counter()
val, (d0, d1) = autodiff(gram, k, X, Y)(params, vector)
val.block_until_ready()
d0.block_until_ready()
d1.block_until_ready()
print("\tRun time:", time.perf_counter() - t0)
