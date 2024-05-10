"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

import argparse
import os
import time
from typing import Callable

import gpytorch.kernels
import gpytorch.kernels.keops
import jax
import jax.numpy as jnp
import torch
from matfree_extensions.util import exp_util, gp_util


def print_ts(ts: jax.Array, label_: str, /, *, num_runs: int):
    ts = jnp.asarray(ts)
    amin, median, amax = jnp.amin(ts), jnp.median(ts), jnp.amax(ts)
    msg = f"{amin:.1e} < {median:.1e} < {amax:.1e}"
    description = f"| minimum < median < maximum of {num_runs} runs | {label_}"
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
    kernel = gpytorch.kernels.keops.RBFKernel()
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    return time_matvec(lambda v, p: kernel(p) @ v, vec, x, num_runs=num_runs)


def time_matfree(prng_seed, N: int, shape_in: tuple, mv, *, num_runs: int):
    prng_key = jax.random.PRNGKey(prng_seed)
    key1, key2 = jax.random.split(prng_key)
    x = jax.random.normal(key1, shape=(N, *shape_in))
    vec = jax.random.normal(key1, shape=(N,))
    kernel, params = gp_util.kernel_scaled_rbf(shape_in=shape_in, shape_out=())
    fun = jax.jit(mv(kernel(**params)))

    def matvec_fun(v, p):
        return fun(p, p, v).block_until_ready()

    return time_matvec(matvec_fun, vec, x, num_runs=num_runs)


if __name__ == "__main__":
    # todo: data_dim -> log_data_dim to go up to 100+
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--log_data_size", type=int, required=True)
    parser.add_argument("--data_dim", type=int, required=True)
    args = parser.parse_args()

    # Translate the argparse into a filename
    title = "matvec"
    title += f"_num_runs_{args.num_runs}"
    title += f"_data_size_{2**args.log_data_size}"
    title += f"_data_dim_{args.data_dim}"

    # Parameter setup
    print(f"\nParameter setup: N = {2**args.log_data_size}, d = {args.data_dim}")
    seed = args.data_dim  # Use the current "size" as a seed
    params = (2**args.log_data_size, (args.data_dim,))
    print("--------------------------------")

    # Start the simulation
    results: dict[str, jax.Array] = {}

    if 2**args.log_data_size <= 200_000:
        label = "matfree_map"
        matvec = gp_util.gram_matvec_map(checkpoint=True)
        t = time_matfree(seed, *params, mv=matvec, num_runs=args.num_runs)
        print_ts(t, label, num_runs=args.num_runs)
        results[label] = t

    label = "matfree_vmap"
    num_batches = int(jnp.minimum(128, 2**args.log_data_size))
    matvec = gp_util.gram_matvec_map_over_batch(
        num_batches=num_batches, checkpoint=True
    )
    # matvec = gp_util.gram_matvec_full_batch()
    t = time_matfree(seed, *params, mv=matvec, num_runs=args.num_runs)
    print_ts(t, label, num_runs=args.num_runs)
    results[label] = t

    label = "gpytorch"
    t = time_gpytorch_via_pykeops(seed, *params, num_runs=args.num_runs)
    print_ts(t, label, num_runs=args.num_runs)
    results[label] = t

    print()

    print("Saving to a file")
    directory = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory, exist_ok=True)

    for name, value in results.items():
        path = f"{directory}/{title}_{name}.npy"
        jnp.save(path, jnp.asarray(value))
