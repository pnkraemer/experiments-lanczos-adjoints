"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

# todo: plot the results

# todo: collect memory consumption

# todo: inside a jit, and on GPU/CPU,
#  is calling a map() instead of vmap() really necessary?

# todo: understand how to benchmark memory in torch

# todo: submit as a script
#  (I think the memory stats might be contaminated by other processes)

# todo: assert values are identical

# todo: is broadcasting faster than vmap?

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
    msg = f"{amin:.1e} < {median:.1e} < {amax:.1e}"
    description = f"| min < med < max of {num_runs} runs | {label}"
    print(msg, description)


def time_matvec(mv: Callable, vec: jax.Array, params, *, num_runs: int):
    ts = []
    _ = mv(vec, params)  # for potential pre-compilation
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = mv(vec, params)
        t1 = time.perf_counter()

        ts.append(t1 - t0)
    return jnp.asarray(ts)


def time_gpytorch_via_pykeops(prng_seed, N: int, shape_in: tuple, *, num_runs: int):
    torch.manual_seed(prng_seed)
    x = torch.randn((N, *shape_in))
    vec = torch.randn((N,))
    kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5)
    return time_matvec(lambda v, p: kernel(p) @ v, vec, x, num_runs=num_runs)


def time_matfree(prng_seed, N: int, shape_in: tuple, mv, *, num_runs: int):
    key = jax.random.PRNGKey(prng_seed)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, shape=(N, *shape_in))
    vec = jax.random.normal(key1, shape=(N,))
    kernel, params = gp.kernel_scaled_matern_32(shape_in=(), shape_out=())
    fun = jax.jit(mv(kernel(**params)))

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
    num_runs = args.num_runs

    matrix_sizes = 2**powers
    for idx, num in zip(powers, matrix_sizes):
        # Use the current "size" as a seed
        seed = idx

        print(f"\nI = {idx}, N = {num}")
        print("------------------")

        gpytorch_label = "GPyTorch (via pykeops)"
        gpytorch_t = time_gpytorch_via_pykeops(seed, num, (), num_runs=num_runs)
        print_ts(gpytorch_t, label=gpytorch_label, num_runs=num_runs)

        map_label = "Matfree (via JAX's map)"
        map_matvec = gp.gram_matvec_map()
        map_t = time_matfree(seed, num, (), mv=map_matvec, num_runs=num_runs)
        print_ts(map_t, label=map_label, num_runs=num_runs)

        if num >= 16:
            b16_label = "Matfree (via map-over-vmap; 16)"
            b16_matvec = gp.gram_matvec_map_over_batch(batch_size=16)
            b16_t = time_matfree(seed, num, (), mv=b16_matvec, num_runs=num_runs)
            print_ts(b16_t, label=b16_label, num_runs=num_runs)

        if num >= 256:
            b256_label = "Matfree (via map-over-vmap; 256)"
            b256_matvec = gp.gram_matvec_map_over_batch(batch_size=256)
            b256_t = time_matfree(seed, num, (), mv=b256_matvec, num_runs=num_runs)
            print_ts(b256_t, label=b256_label, num_runs=num_runs)

        if num >= 4096:
            b4096_label = "Matfree (via map-over-vmap; 4096)"
            b4096_matvec = gp.gram_matvec_map_over_batch(batch_size=4096)
            b4096_t = time_matfree(seed, num, (), mv=b4096_matvec, num_runs=num_runs)
            print_ts(b4096_t, label=b4096_label, num_runs=num_runs)

        # 8 GB memory allows storing at most 44_000 rows/columns,
        # but the process gets killed around 30_000 already
        if num <= 30_000:
            vmap_label = "Matfree (via JAX's vmap)"
            vmap_matvec = gp.gram_matvec_dense()
            vmap_t = time_matfree(seed, num, (), mv=vmap_matvec, num_runs=num_runs)
            print_ts(vmap_t, label=vmap_label, num_runs=num_runs)

        print()
