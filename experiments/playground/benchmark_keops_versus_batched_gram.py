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

import gpytorch.kernels
import gpytorch.kernels.keops
import jax
import jax.numpy as jnp
import torch
from matfree_extensions import gp


def print_ts(t: jax.Array, *, label: str, num_matvecs: int):
    t = jnp.asarray(t)
    m, s = jnp.mean(t), jnp.std(t)
    print(f"{label}:\t{m:.3f} +/- {s:.3f} sec (mean +/- std of {num_matvecs} runs)")


def run(N, *, num_matvecs):
    shape_in = ()
    x = torch.randn((N, *shape_in))
    v = torch.randn((N,))

    kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5)
    K = kernel(x)

    ts = []
    for _ in range(num_matvecs):
        t0 = time.perf_counter()
        _ = K @ v
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    ts_torch = jnp.asarray(ts)

    kernel, params = gp.kernel_scaled_matern_32(shape_in=(), shape_out=())
    matvec = gp.gram_matvec_map()
    K = jax.jit(matvec(kernel(**params)))
    x_ = jnp.asarray(x.detach().numpy())
    v_ = jnp.asarray(v.detach().numpy())
    _ = K(x_, x_, v_)  # pre-compile

    ts = []
    for _ in range(num_matvecs):
        t0 = time.perf_counter()
        vv = K(x_, x_, v_)
        vv.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    ts_matfree = jnp.asarray(ts)
    return ts_torch, ts_matfree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_matvecs", type=int, required=True)
    parser.add_argument("--matrix_size_min", type=int, required=True)
    parser.add_argument("--matrix_size_max", type=int, required=True)
    args = parser.parse_args()

    powers = jnp.arange(args.matrix_size_min, args.matrix_size_max)
    matrix_sizes = 2**powers
    for idx, num in zip(powers, matrix_sizes):
        t_torch, t_matfree = run(num, num_matvecs=args.num_matvecs)
        print(f"\nI = {idx}, N = {num}")
        print("------------------")
        print_ts(t_torch, label="Keops", num_matvecs=args.num_matvecs)
        print_ts(t_matfree, label="JAX", num_matvecs=args.num_matvecs)
        print()
