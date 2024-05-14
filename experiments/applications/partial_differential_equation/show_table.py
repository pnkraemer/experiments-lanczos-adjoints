import argparse
import pickle

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import pandas as pd
from matfree_extensions.util import exp_util

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
args = parser.parse_args()
print(args)


labels = {
    "arnoldi": r"Arnoldi \& Adjoints (ours)",
    "diffrax:dopri5+backsolve": r"Dopri5 \& Backsolve ",
    "diffrax:tsit5+recursive_checkpoint": r"Tsit5 \& RecursiveCheckpoint",
}
methods = list(labels.keys())


directory = exp_util.matching_directory(__file__, "results/")

stats_mean = {}
stats_std = {}
for method in methods:
    losses = []
    rmses = []
    runtimes = []

    for seed in [1, 2, 3]:
        path = f"{directory}{args.resolution}x{args.resolution}_{method}_s{seed}"
        with open(f"{path}_stats.pkl", "rb") as handle:
            results = pickle.load(handle)

        timestamps = jnp.load(f"{path}_timestamps.npy")

        losses.append(results["loss"])
        rmses.append(results["rmse_param"])
        runtimes.append(timestamps[-1] / len(timestamps))

    losses = jnp.asarray(losses)
    rmses = jnp.asarray(rmses)
    runtimes = jnp.asarray(runtimes)

    stats_mean[labels[method]] = {
        "Loss on test set": jnp.mean(losses),
        "Parameter RMSE": jnp.mean(rmses),
        "Runtime per epoch": jnp.mean(runtimes),
    }
    print(losses)
    print(jnp.std(losses))
    stats_std[labels[method]] = {
        "Loss on test set": jnp.std(losses),
        "Parameter RMSE": jnp.std(rmses),
        "Runtime per epoch": jnp.std(runtimes),
    }


stats_mean = jax.tree_util.tree_map(float, stats_mean)
stats_std = jax.tree_util.tree_map(float, stats_std)

results_dataframe_mean = pd.DataFrame(stats_mean)
results_dataframe_std = pd.DataFrame(stats_std)

print()
print()
print()
print()
print()

latex = results_dataframe_mean.to_latex(float_format="%.1e")
print(latex)
print()
print()
print()
print()
print()
latex = results_dataframe_std.to_latex(float_format="%.1e")
print(latex)
print()
print()
print()
print()
print()
