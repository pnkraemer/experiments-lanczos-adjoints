"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

import argparse
import functools
import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
from matfree import hutchinson
from matfree_extensions.util import exp_util, gp_util


def print_ts(t: jax.Array, *, label: str, num_runs: int):
    t = jnp.asarray(t)
    amin, median, amax = jnp.amin(t), jnp.median(t), jnp.amax(t)
    msg = f"{amin:.1e} < {median:.1e} < {amax:.1e}"
    description = f"| min < med < max of {num_runs} runs | {label}"
    print(msg, description)


def time_gp_mll(
    mv: Callable,
    v0: jax.Array,
    input_dim: int,
    *,
    num_runs,
    num_batches=1,
    num_samples=1,
    krylov_depth=10,
    cg_tol=1.0,
    checkpoint_montecarlo: bool,
):
    sampler = hutchinson.sampler_rademacher(v0, num=num_samples)
    logpdf = gp_util.logpdf_lanczos(
        krylov_depth,
        sampler,
        slq_batch_num=num_batches,
        cg_tol=cg_tol,
        checkpoint=checkpoint_montecarlo,
    )

    k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(input_dim,), shape_out=())

    prior = gp_util.model(gp_util.mean_zero(), k, gram_matvec=mv)
    likelihood, p_likelihood = gp_util.likelihood_gaussian()
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf)

    xs = jnp.linspace(0, 1, num=len(v0), endpoint=True)
    xs = jnp.stack([xs] * input_dim, axis=1)
    ys = jnp.linspace(0, 1, num=len(v0), endpoint=True)

    key = jax.random.PRNGKey(1)

    p1_flat, unflatten_1 = jax.flatten_util.ravel_pytree(p_prior)
    p2_flat, unflatten_2 = jax.flatten_util.ravel_pytree(p_likelihood)

    @jax.jit
    @functools.partial(jax.value_and_grad, has_aux=True)
    def fun(p1, p2):
        return loss(
            xs, ys, key, params_prior=unflatten_1(p1), params_likelihood=unflatten_2(p2)
        )

    (_value, _aux), _grad = fun(p1_flat, p2_flat)

    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        (val, _aux), grad = fun(p1_flat, p2_flat)
        val.block_until_ready()
        grad.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return jnp.asarray(ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, required=True)
    parser.add_argument("--log_data_size", type=int, required=True)
    parser.add_argument("--data_dim", type=int, required=True)
    parser.add_argument("--checkpoint_matvec", action="store_true")
    parser.add_argument("--checkpoint_montecarlo", action="store_true")
    args = parser.parse_args()
    print(args)

    title = "mll"
    title += f"_num_runs_{args.num_runs}"
    title += f"_data_size_{2**args.log_data_size}"
    title += f"_data_dim_{args.data_dim}"
    if args.checkpoint_matvec:
        title += "checkpoint_matvec"
    if args.checkpoint_montecarlo:
        title += "_checkpoint_montecarlo"

    # Use the current "size" as a seed
    seed = args.log_data_size

    num = 2**args.log_data_size
    results: dict[str, jax.Array] = {}

    print(f"\nN = {num}")
    print("------------------")

    vec = jnp.ones((num,), dtype=float)

    label = "matfree_vmap"
    matvec = gp_util.gram_matvec_full_batch()
    t = time_gp_mll(
        matvec,
        vec,
        args.data_dim,
        num_runs=args.num_runs,
        checkpoint_montecarlo=args.checkpoint_montecarlo,
    )
    print_ts(t, label=label, num_runs=args.num_runs)
    results[label] = t

    label = "matfree_map"
    matvec = gp_util.gram_matvec_map(checkpoint=args.checkpoint_matvec)
    t = time_gp_mll(
        matvec,
        vec,
        args.data_dim,
        num_runs=args.num_runs,
        checkpoint_montecarlo=args.checkpoint_montecarlo,
    )
    print_ts(t, label=label, num_runs=args.num_runs)
    results[label] = t

    print("\nSaving to a file")
    directory = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory, exist_ok=True)

    for name, value in results.items():
        path = f"{directory}/{title}_{name}.npy"
        jnp.save(path, jnp.asarray(value))
