"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

# todo: plot the results

# todo: collect memory consumption

# todo: understand how to benchmark memory in torch

# todo: submit as a script
#  (I think the memory stats might be contaminated by other processes)

import argparse
import time
from typing import Callable

import gpytorch.kernels
import gpytorch.kernels.keops
import jax
import jax.numpy as jnp
import torch
from matfree import hutchinson
from matfree_extensions.util import gp_util


def print_ts(t: jax.Array, *, label: str, num_runs: int):
    t = jnp.asarray(t)
    amin, median, amax = jnp.amin(t), jnp.median(t), jnp.amax(t)
    msg = f"{amin:.1e} < {median:.1e} < {amax:.1e}"
    description = f"| min < med < max of {num_runs} runs | {label}"
    print(msg, description)


def time_gp_mll(mv: Callable, vec: jax.Array, *, num_runs, num_batches=10, num_samples=1, krylov_depth=100):
    sampler = hutchinson.sampler_rademacher(vec, num=num_samples)
    logpdf = gp_util.logpdf_lanczos(krylov_depth, sampler, slq_batch_num=num_batches)



    k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=())
    prior = gp_util.model(gp_util.mean_zero(), k, gram_matvec=mv)
    likelihood, p_likelihood = gp_util.likelihood_gaussian()
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf)

    xs = jnp.linspace(0, 1, num=len(vec), endpoint=True)
    ys = xs     
    key = jax.random.PRNGKey(1)
    
    p1_flat, unflatten_1 = jax.flatten_util.ravel_pytree(p_prior)
    p2_flat, unflatten_2 = jax.flatten_util.ravel_pytree(p_likelihood)

    # Change this fun to value_and_grad, and memory blows up.
    # I think it is because of a JVP of the operation (v \mapsto Kv),
    # which accidentally assembles a dense matrix and asks for N^2 memory.

    @jax.jit
    def fun(p1, p2):
        return loss(xs, ys, key, params_prior=unflatten_1(p1), params_likelihood=unflatten_2(p2))
    
    _value = fun(p1_flat, p2_flat)

    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        value = fun(p1_flat, p2_flat)
        value.block_until_ready()
        # grad.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return jnp.asarray(ts)


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

        vec = jnp.ones((num,), dtype=float)

        map_label = "Matfree (via JAX's map)"
        map_matvec = gp_util.gram_matvec_map()
        map_t = time_gp_mll(map_matvec, vec, num_runs=num_runs)
        print_ts(map_t, label=map_label, num_runs=num_runs)

        if num >= 16:
            b16_label = "Matfree (via map-over-vmap; 16)"
            b16_matvec = gp_util.gram_matvec_map_over_batch(batch_size=16)
            b16_t = time_gp_mll(b16_matvec, vec, num_runs=num_runs)
            print_ts(b16_t, label=b16_label, num_runs=num_runs)

        if num >= 256:
            b256_label = "Matfree (via map-over-vmap; 256)"
            b256_matvec = gp_util.gram_matvec_map_over_batch(batch_size=256)
            b256_t = time_gp_mll(b256_matvec, vec, num_runs=num_runs)
            print_ts(b256_t, label=b256_label, num_runs=num_runs)

        if num >= 4096:
            b4096_label = "Matfree (via map-over-vmap; 4096)"
            b4096_matvec = gp_util.gram_matvec_map_over_batch(batch_size=4096)
            b4096_t = time_gp_mll(b4096_matvec, vec, num_runs=num_runs)
            print_ts(b4096_t, label=b4096_label, num_runs=num_runs)

        # 8 GB memory allows storing at most 44_000 rows/columns,
        # but the process gets killed around 30_000 already
        if num <= 30_000:
            vmap_label = "Matfree (via JAX's vmap)"
            vmap_matvec = gp_util.gram_matvec_full_batch()
            vmap_t = time_gp_mll(vmap_matvec, vec, num_runs=num_runs)
            print_ts(vmap_t, label=vmap_label, num_runs=num_runs)

        print()
