"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

# todo: plot the results
# todo: collect memory consumption
# todo: inside a jit, and on GPU/CPU,
#  is calling a map() instead of vmap() really necessary?
# todo: understand how to benchmark memory in torch
# todo: matvec_gram_batches poorly:
#  Instead of (vmap(k)(x,y)) @ v, it should call vmap(k(-,y) @ v)(x)
# todo: submit as a script
#  (I think the memory stats might be contaminated by other processes)
# todo: assert values are identical

import argparse
import time
from typing import Callable

import gpytorch.kernels
import gpytorch.kernels.keops
import jax
import jax.numpy as jnp
import torch
from matfree_extensions import gp


def print_ts(t: jax.Array, *, label: str, num_runs: int):
    t = jnp.asarray(t)
    amin, median, amax = jnp.amin(t), jnp.median(t), jnp.amax(t)
    print(f"{label}: \t{amin:.3f} < {median:.3f} < {amax:.3f} of {num_runs} runs")


def time_matvec(matvec: Callable, vec: jax.Array, params, *, num_runs: int):
    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = matvec(vec, params)
        t1 = time.perf_counter()

        ts.append(t1 - t0)
    return jnp.asarray(ts)


def time_gpytorch_via_pykeops(prng_seed, N: int, shape_in: tuple, *, num_runs: int):
    torch.manual_seed(prng_seed)
    x = torch.randn((N, *shape_in))
    vec = torch.randn((N,))
    kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5)
    return time_matvec(lambda v, p: kernel(p) @ v, vec, x, num_runs=num_runs)


def time_matfree(prng_seed, N: int, shape_in: tuple, matvec, *, num_runs: int):
    key = jax.random.PRNGKey(prng_seed)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, shape=(N, *shape_in))
    vec = jax.random.normal(key1, shape=(N,))
    kernel, params = gp.kernel_scaled_matern_32(shape_in=(), shape_out=())
    fun = jax.jit(matvec(kernel(**params)))

    def matvec_fun(v, p):
        return fun(p, p, v).block_until_ready()

    return time_matvec(matvec_fun, vec, x, num_runs=num_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--matrix_size_min", type=int, required=True)
    parser.add_argument("--matrix_size_max", type=int, required=True)
    args = parser.parse_args()

    powers = jnp.arange(args.matrix_size_min, args.matrix_size_max)
    matrix_sizes = 2**powers
    for idx, num in zip(powers, matrix_sizes):
        # Use the current "size" as a seed
        seed = idx

        matvec_map = gp.gram_matvec_map()
        t_map = time_matfree(seed, num, (), matvec=matvec_map, num_runs=args.num_runs)

        matvec_vmap = gp.gram_matvec_dense()
        t_vmap = time_matfree(seed, num, (), matvec=matvec_vmap, num_runs=args.num_runs)

        t_gpytorch = time_gpytorch_via_pykeops(seed, num, (), num_runs=args.num_runs)

        print(f"\nI = {idx}, N = {num}")
        print("------------------")
        print_ts(t_gpytorch, label="GPyTorch (via pykeops) ", num_runs=args.num_runs)
        print_ts(t_map, label="Matfree (via JAX's map)", num_runs=args.num_runs)
        print_ts(t_vmap, label="Matfree (via JAX's vmap)", num_runs=args.num_runs)
        print()
