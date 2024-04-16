"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

import time

import gpytorch.kernels
import gpytorch.kernels.keops  # requires pip install pykeops
import jax
import jax.numpy as jnp
import torch
from matfree_extensions import gp

num_matvecs = 3


def print_ts(t):
    t = jnp.asarray(t)
    print(
        f"\t{jnp.mean(t):.3f} +/- {jnp.std(t):.3f} "
        f"(mean +/- std of {num_matvecs} runs; all in seconds)"
    )


for num in 2 ** jnp.arange(10, 16):
    print(f"\nN = {num}")
    print("-----------")

    shape_in = ()
    x = torch.randn((num, *shape_in))
    v = torch.randn((num,))

    print("GPyTorch + Keops (Matern32)")
    kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5)
    K = kernel(x)

    ts = []
    for _ in range(num_matvecs):
        t0 = time.perf_counter()
        _ = K @ v
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    print_ts(ts)

    print("Matfree (Matern32)")

    kernel, params = gp.kernel_scaled_matern_32(shape_in=(), shape_out=())
    matvec = gp.gram_matvec_map_over_batch(batch_size=16)
    K = jax.jit(matvec(kernel(**params)))
    x_ = jnp.asarray(x.detach().numpy())
    v_ = jnp.asarray(v.detach().numpy())
    _ = K(x_, x_, v_)

    ts = []
    for _ in range(num_matvecs):
        t0 = time.perf_counter()
        vv = K(x_, x_, v_)
        vv.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    print_ts(ts)
    print("\n")
