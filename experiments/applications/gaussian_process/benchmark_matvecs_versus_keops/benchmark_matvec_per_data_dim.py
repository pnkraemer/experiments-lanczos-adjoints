"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

# todo: plot the results

# todo: collect memory consumption

# todo: understand how to benchmark memory in torch

# todo: submit as a script
#  (I think the memory stats might be contaminated by other processes)

import os
import argparse
import time
from typing import Callable

import gpytorch.kernels
import gpytorch.kernels.keops
import jax
import jax.numpy as jnp
import torch
from matfree_extensions.util import gp_util, exp_util


def print_ts(t: jax.Array, *, label: str, num_runs: int):
    t = jnp.asarray(t)
    amin, median, amax = jnp.amin(t), jnp.median(t), jnp.amax(t)
    msg = f"{amin:.1e} < {median:.1e} < {amax:.1e}"
    description = f"| minimum < median < maximum of {num_runs} runs | {label}"
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
    x = torch.randn((N, *shape_in)).cuda()
    vec = torch.randn((N,)).cuda()
    kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5)
    kernel = gpytorch.kernels.ScaleKernel(kernel).cuda()
    return time_matvec(lambda v, p: kernel(p) @ v, vec, x, num_runs=num_runs)


def time_matfree(prng_seed, N: int, shape_in: tuple, mv, *, num_runs: int):
    key = jax.random.PRNGKey(prng_seed)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, shape=(N, *shape_in))
    vec = jax.random.normal(key1, shape=(N,))
    kernel, params = gp_util.kernel_scaled_matern_32(shape_in=shape_in, shape_out=())
    fun = jax.jit(mv(kernel(**params)))

    def matvec_fun(v, p):
        return fun(p, p, v).block_until_ready()

    return time_matvec(matvec_fun, vec, x, num_runs=num_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--data_dim_min", type=int, required=True)
    parser.add_argument("--data_dim_max", type=int, required=True)
    parser.add_argument("--data_size", type=int, default=40_000)
    args = parser.parse_args()

    results = {}

    for dim in range(args.data_dim_min, args.data_dim_max):
        # Use the current "size" as a seed
        seed = dim

        print(f"\ndim = {dim}")
        print("------------------")
        if not "dim" in results:
            results["dim"] = []
        results["dim"].append(dim)


        label = "Matfree (via JAX's map)"
        matvec = gp_util.gram_matvec_map()
        t = time_matfree(seed, args.data_size, (dim,), mv=matvec, num_runs=args.num_runs)
        print_ts(t, label=label, num_runs=args.num_runs)
        if not label in results:
            results[label] = []
        results[label].append(t)
            
        for bnum in reversed([1, 10, 200, 4000]):
            if args.data_size >= bnum:
                label = f"Matfree (via map-over-vmap; {bnum} batches)"
                matvec = gp_util.gram_matvec_map_over_batch(num_batches=bnum)
                t = time_matfree(seed, args.data_size, (dim,), mv=matvec, num_runs=args.num_runs)
                print_ts(t, label=label, num_runs=args.num_runs)
                if not label in results:
                    results[label] = []
                results[label].append(t)
                
        label = "Matfree (via JAX's vmap)"
        matvec = gp_util.gram_matvec_full_batch()
        t = time_matfree(seed, args.data_size, (dim,), mv=matvec, num_runs=args.num_runs)
        print_ts(t, label=label, num_runs=args.num_runs)
        if not label in results:
            results[label] = []
        results[label].append(t)
            
        label = "GPyTorch (via pykeops)"
        t = time_gpytorch_via_pykeops(seed, args.data_size, (dim,), num_runs=args.num_runs)
        print_ts(t, label=label, num_runs=args.num_runs)
        if not label in results:
            results[label] = []
        results[label].append(t)
        
        print()

print("Saving to a file")
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)
LABEL = f"num_runs_{args.num_runs}_data_size_{args.data_size}"

for key, value in results.items():
    path = f"{directory}/matvec_per_data_dim_{LABEL}_{key}.npy"
    jnp.save(path, jnp.asarray(value))
