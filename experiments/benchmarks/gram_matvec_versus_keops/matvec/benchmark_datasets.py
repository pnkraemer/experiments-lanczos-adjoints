"""Compare the efficiency of GPyTorch+KeOps to gp.gram_matvec_map()."""

# todo: plot the results

# todo: collect memory consumption

# todo: understand how to benchmark memory in torch

# todo: submit as a script
#  (I think the memory stats might be contaminated by other processes)

import argparse
import os
import time
from typing import Callable

import gpytorch.kernels
import gpytorch.kernels.keops
import jax
import jax.numpy as jnp
import torch
from matfree_extensions.util import exp_util, gp_util, uci_util

DATASET_LIST = [
    "road",
    "song",
    "air_quality",
    "bike",
    "kegg",
    "parkinson",
    "protein",
    "sgemm",
    "concrete",
    "power_plant",
]
# DATASET_LIST = ["protein", "sgemm", "concrete", "power_plant"]


def get_uci_data(label):
    if label == "road":
        print("\nRoad network:")
        inputs, targets = uci_util.uci_road_network()
        n_train = 350_000

        print(inputs.shape, targets.shape)

    elif label == "song":
        print("\nSong:")
        inputs, targets = uci_util.uci_song()
        n_train = 500_000
        print(inputs.shape, targets.shape)

    elif label == "air_quality":
        print("\nAir quality:")
        inputs, targets = uci_util.uci_air_quality()
        n_train = 300_000
        print(inputs.shape, targets.shape)

    elif label == "bike":
        print("\nBike sharing:")
        inputs, targets = uci_util.uci_bike_sharing()
        n_train = 15_000
        print(inputs.shape, targets.shape)

    elif label == "kegg":
        print("\nKEGG undirected:")
        inputs, targets = uci_util.uci_kegg_undirected()
        n_train = 50_000
        print(inputs.shape, targets.shape)

    elif label == "parkinson":
        print("\nParkinson:")
        inputs, targets = uci_util.uci_parkinson()
        n_train = 5_000
        print(inputs.shape, targets.shape)

    elif label == "protein":
        print("\nProtein:")
        inputs, targets = uci_util.uci_protein()
        n_train = 45_000
        print(inputs.shape, targets.shape)

    elif label == "sgemm":
        print("\nSGEMM:")
        inputs, targets = uci_util.uci_sgemm()
        n_train = 200_000
        print(inputs.shape, targets.shape)

    elif label == "concrete":
        print("\nConcrete:")
        inputs, targets = uci_util.uci_concrete()
        n_train = 1_000
        print(inputs.shape, targets.shape)

    elif label == "power_plant":
        print("\nPower plant:")
        inputs, targets = uci_util.uci_power_plant()
        n_train = 8_000
        print(inputs.shape, targets.shape)

    else:
        print("\nNot in the list!")
        return -1, None

    return inputs[:n_train], targets[:n_train]


def print_ts(ts: jax.Array, label_: str, /, *, num_runs: int):
    ts = jnp.asarray(ts)
    amin, median, amax = jnp.amin(ts), jnp.median(ts), jnp.amax(ts)
    mean, std = jnp.mean(ts), jnp.std(ts)
    msg_1 = f"{amin:.3e} < {median:.3e} < {amax:.3e}"
    msg_2 = f"{mean:.3e} +/- {std:.3e}"
    description_1 = f"| minimum < median < maximum of {num_runs} runs | {label_}"
    description_2 = f"| mean +/- std in {num_runs} runs | {label_}"
    print(msg_1, description_1)
    print(msg_2, description_2)


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
    parser.add_argument("--num_runs", type=int, default=5)
    args = parser.parse_args()

    for label in DATASET_LIST:
        inputs, targets = get_uci_data(label)

        # Translate the argparse into a filename
        title = "matvec"
        title += f"_num_runs_{args.num_runs}"
        title += f"_data_set_{label}"

        size = inputs.shape[0]
        dim = inputs.shape[1]

        # Parameter setup
        print(f"\nParameter setup: N = {size}, data-dim = {dim}")
        print("--------------------------------")

        # Start the simulation
        results: dict[str, jax.Array] = {}

        seed = size  # Use the current "dim" as a seed
        params = (size, (dim,))

        if size <= 300_000:
            label = "matfree_sequential"
            matvec = gp_util.gram_matvec_sequential(checkpoint=True)
            t = time_matfree(seed, *params, mv=matvec, num_runs=args.num_runs)
            print_ts(t, label, num_runs=args.num_runs)
            results[label] = t

        if size <= 40_000:
            label = "matfree_full"
            matvec = gp_util.gram_matvec()
            t = time_matfree(seed, *params, mv=matvec, num_runs=args.num_runs)
            print_ts(t, label, num_runs=args.num_runs)
            results[label] = t

        if size <= 100_000:
            label = "matfree_partitioned_10"
            num_batches = int(jnp.minimum(10, size))
            matvec = gp_util.gram_matvec_partitioned(num=num_batches, checkpoint=True)
            t = time_matfree(seed, *params, mv=matvec, num_runs=args.num_runs)
            print_ts(t, label, num_runs=args.num_runs)
            results[label] = t

        if size <= 300_000:
            label = "matfree_partitioned_100"
            num_batches = int(jnp.minimum(100, size))
            matvec = gp_util.gram_matvec_partitioned(num=num_batches, checkpoint=True)
            t = time_matfree(seed, *params, mv=matvec, num_runs=args.num_runs)
            print_ts(t, label, num_runs=args.num_runs)
            results[label] = t

        label = "matfree_partitioned_1000"
        num_batches = int(jnp.minimum(1_000, size))
        matvec = gp_util.gram_matvec_partitioned(num=num_batches, checkpoint=True)
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
