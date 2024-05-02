"""Measure the vjp-wall-time for a sparse matrix."""

import argparse
import functools
import os
import time

import jax.experimental.sparse
import jax.flatten_util
import jax.numpy as jnp
from matfree_extensions import lanczos
from matfree_extensions.util import exp_util

parser = argparse.ArgumentParser()
parser.add_argument("--reortho", type=str, required=True)
parser.add_argument("--precompile", action="store_true")
parser.add_argument("--num_runs", type=int, required=True)
args = parser.parse_args()
print(args)


def _rmse_relative(x, y, /, *, nugget):
    diff_abs = jnp.abs(x - y)
    normalise = nugget + jnp.abs(y)
    return jnp.linalg.norm(diff_abs / normalise) / jnp.size(diff_abs)


# n = 10_000
seed = 1


# Set up a test-matrix
# "t2dal_e" is diagonal, spd, and nicely small (5_000x5_000)
# "bloweybq" is not diagonal, spd, and nicely large
# "t3dl_e" is diagonal, spd, and nicely large (20_000x20_000)
matrix_which = "t3dl_e"
path = "./data/matrices/"
M = exp_util.suite_sparse_load(matrix_which, path=path)
print(M.shape)

params, params_unflatten = jax.flatten_util.ravel_pytree(M.data)


@jax.jit
def matvec(x, p):
    # return p * x
    pp = params_unflatten(p)
    P_ = jax.experimental.sparse.BCOO((pp, M.indices), shape=M.shape)
    return P_ @ x


n = M.shape[0]

# Set up an initial vector
vector = jax.random.normal(jax.random.PRNGKey(seed + 1), shape=(n,))

# Flatten the inputs
flat, unflatten = jax.flatten_util.ravel_pytree((vector, params))

krylov_depths = []
times_fwdpass = []
times_custom = []
times_autodiff = []
norms_of_differences = []

for krylov_depth in jnp.arange(10, 100, step=10):
    krylov_depth = int(krylov_depth)
    print("Krylov-depth:", krylov_depth)
    krylov_depths.append(krylov_depth)

    # Construct a vector-to-vector decomposition function
    def decompose(f, *, custom_vjp):  # todo: resolve noqa
        algorithm = lanczos.tridiag(
            matvec,
            krylov_depth,  # noqa: B023
            custom_vjp=custom_vjp,
            reortho=args.reortho,
        )
        output = algorithm(*unflatten(f))
        return jax.flatten_util.ravel_pytree(output)[0]

    # Construct the two implementations
    reference = jax.jit(functools.partial(decompose, custom_vjp=False))
    implementation = jax.jit(functools.partial(decompose, custom_vjp=True))

    # Compute both VJPs

    # Compute a VJP into a random direction
    # (This ignores potential symmetry/orthogonality constraints of the outputs.
    # But we only care about speed at this point, so it is fine.)
    key = jax.random.PRNGKey(seed + 2)
    dnu = jax.random.normal(key, shape=jnp.shape(reference(flat)))

    fx_imp, vjp_imp = jax.vjp(implementation, flat)

    if args.precompile:
        _ = implementation(flat).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(args.num_runs):
        _ = implementation(flat).block_until_ready()
    t1 = time.perf_counter()
    time_fwdpass = (t1 - t0) / args.num_runs
    times_fwdpass.append(time_fwdpass)
    print("Time (forward pass):\n\t", time_fwdpass)

    vjp_imp = jax.jit(vjp_imp)
    if args.precompile:
        _ = vjp_imp(dnu)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(args.num_runs):
        _ = vjp_imp(dnu)[0].block_until_ready()
    t1 = time.perf_counter()
    time_custom = (t1 - t0) / args.num_runs
    times_custom.append(time_custom)
    print("Time (custom VJP):\n\t", time_custom)

    if krylov_depth < 50:
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

        # diff = vjp_ref(dnu)[0] - vjp_imp(dnu)[0]
        diff = _rmse_relative(vjp_imp(dnu)[0], vjp_ref(dnu)[0], nugget=0.0)
        norms_of_differences.append(diff)
        print("Norm of VJP-difference:\n\t", diff)
        print(
            "Ratio of VJP run times (small is good):\n\t", time_custom / time_autodiff
        )

    print()


directory = exp_util.matching_directory(__file__, "data/")
os.makedirs(directory, exist_ok=True)
jnp.save(f"{directory}/{matrix_which}_krylov_depths.npy", jnp.asarray(krylov_depths))
jnp.save(f"{directory}/{matrix_which}_times_fwdpass.npy", jnp.asarray(times_fwdpass))
jnp.save(f"{directory}/{matrix_which}_times_custom.npy", jnp.asarray(times_custom))
jnp.save(f"{directory}/{matrix_which}_times_autodiff.npy", jnp.asarray(times_autodiff))
jnp.save(
    f"{directory}/{matrix_which}_norms_of_differences.npy",
    jnp.asarray(norms_of_differences),
)
