"""Measure the vjp-wall-time for a sparse matrix."""

import argparse
import os
import time

import jax.experimental.sparse
import jax.flatten_util
import jax.numpy as jnp
from matfree_extensions import arnoldi, lanczos
from matfree_extensions.util import exp_util

# How to set up a test-matrix:
#   Debug with "1138_bus" (small)
#   Run with "gyro" (like 1138_bus, but 20,000 rows and 1 mio params.)
parser = argparse.ArgumentParser()
parser.add_argument("--lanczos_or_arnoldi", type=str, required=True)
parser.add_argument("--reortho", type=str, required=True)
parser.add_argument("--which_matrix", type=str, default="1138_bus")
parser.add_argument("--num_runs", type=int, default=3)
parser.add_argument("--max_krylov_depth", type=int, default=50)
parser.add_argument("--backprop_until", type=int, default=50)
parser.add_argument("--precompile", action="store_true")
args = parser.parse_args()
print(args)

# Label the run (when saving to a file)
LABEL = f"{args.lanczos_or_arnoldi}"
LABEL += f"_{args.which_matrix}"
LABEL += f"_reortho_{args.reortho}"
LABEL += f"_num_runs_{args.num_runs}"
LABEL += f"_backprop_until_{args.backprop_until}"
LABEL += f"_max_krylov_depth_{args.max_krylov_depth}"
LABEL += f"_precompile_{args.precompile}"
print("Label:", LABEL)


def decomposition(mv, /, unflatten_fun, reortho, method):
    def make_decomp(kdepth, /, *, custom_vjp):
        match_methods = {
            "arnoldi": arnoldi.hessenberg(
                mv, kdepth, custom_vjp=custom_vjp, reortho=reortho
            ),
            "lanczos": lanczos.tridiag(
                mv, kdepth, custom_vjp=custom_vjp, reortho=reortho
            ),
        }
        algorithm = match_methods[method]

        @jax.jit
        def decompose(f):
            output = algorithm(*unflatten_fun(f))
            return jax.flatten_util.ravel_pytree(output)[0]

        return decompose

    return make_decomp


path = "./data/matrices/"
# M = exp_util.suite_sparse_download(args.which_matrix, path=path)
M = exp_util.suite_sparse_load(args.which_matrix, path=path)

params, params_unflatten = jax.flatten_util.ravel_pytree(M.data)


@jax.jit
def matvec(x, p):
    pp = params_unflatten(p)
    matrix = jax.experimental.sparse.BCOO((pp, M.indices), shape=M.shape)
    return matrix @ x


# Set up an initial vector and learn how to (un)flatten parameters
n = M.shape[0]
vector = jax.random.normal(jax.random.PRNGKey(1), shape=(n,))
flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))
make = decomposition(matvec, unflatten, args.reortho, args.lanczos_or_arnoldi)

# Start looping
times_fwdpass = []
times_custom = []
times_autodiff = []

n = args.max_krylov_depth // 10
krylov_depths = jnp.arange(n, args.max_krylov_depth + n, step=n, dtype=int)
for krylov_depth in krylov_depths:
    print("Krylov-depth:", krylov_depth)

    # Array(dtype=int) would not be static, so we transform
    krylov_depth = int(krylov_depth)

    # Construct the two implementations
    reference = jax.jit(make(krylov_depth, custom_vjp=False))
    implementation = jax.jit(make(krylov_depth, custom_vjp=True))

    # Compute a VJP into a random direction
    key = jax.random.PRNGKey(krylov_depth)
    dnu = jax.random.normal(key, shape=jnp.shape(reference(flat)))

    print("Evaluating the forward pass")
    if args.precompile:
        _ = implementation(flat).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(args.num_runs):
        _ = implementation(flat).block_until_ready()
    t1 = time.perf_counter()
    time_fwdpass = (t1 - t0) / args.num_runs
    times_fwdpass.append(time_fwdpass)
    print("Time (forward pass):\n\t", time_fwdpass)

    print("Evaluating the adjoint pass")
    fx_imp, vjp_imp = jax.vjp(implementation, flat)
    vjp_imp = jax.jit(vjp_imp)

    if args.precompile:
        _ = vjp_imp(dnu)[0].block_until_ready()

    t0 = time.perf_counter()
    for _ in range(args.num_runs):
        _ = vjp_imp(dnu)[0].block_until_ready()
    t1 = time.perf_counter()
    time_custom = (t1 - t0) / args.num_runs
    times_custom.append(time_custom)
    print("Time (adjoint):\n\t", time_custom)

    if krylov_depth < args.backprop_until:
        print("Evaluating the backprop pass")
        fx_ref, vjp_ref = jax.vjp(reference, flat)
        vjp_ref = jax.jit(vjp_ref)

        if args.precompile:
            _ = vjp_ref(dnu)[0].block_until_ready()

        t0 = time.perf_counter()
        for _ in range(args.num_runs):
            _ = vjp_ref(dnu)[0].block_until_ready()
        t1 = time.perf_counter()
        time_autodiff = (t1 - t0) / args.num_runs
        times_autodiff.append(time_autodiff)
        print("Time (AutoDiff):\n\t", time_autodiff)

        msg = "Ratio of VJP run times (small is good)"
        print(f"{msg}:\n\t", time_custom / time_autodiff)

    print()

print("Saving to a file")
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)

path = f"{directory}/{LABEL}_krylov_depths.npy"
jnp.save(path, jnp.asarray(krylov_depths))

path = f"{directory}/{LABEL}_times_fwdpass.npy"
jnp.save(path, jnp.asarray(times_fwdpass))

path = f"{directory}/{LABEL}_times_custom.npy"
jnp.save(path, jnp.asarray(times_custom))

path = f"{directory}/{LABEL}_times_autodiff.npy"
jnp.save(path, jnp.asarray(times_autodiff))
