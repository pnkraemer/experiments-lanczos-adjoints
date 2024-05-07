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
    "arnoldi": "Arnoldi",
    "diffrax:euler+backsolve": "Euler/Backsolve",
    "diffrax:heun+recursive_checkpoint": "Heun/Autodiff",
    "diffrax:dopri5+backsolve": "Dopri5/Backsolve ",
    "diffrax:tsit5+recursive_checkpoint": "Tsit5/Autodiff",
}
methods = list(labels.keys())


directory = exp_util.matching_directory(__file__, "results/")

stats = {}
for method in methods:
    path = f"{directory}{args.resolution}x{args.resolution}_{method}"
    with open(f"{path}_stats.pkl", "rb") as handle:
        results = pickle.load(handle)

    # jnp.save(f"{path}_parameter.npy", scale_after)
    # jnp.save(f"{path}_matvecs.npy", jnp.asarray(matvecs))
    # jnp.save(f"{path}_convergence.npy", jnp.asarray(convergence))
    timestamps = jnp.load(f"{path}_timestamps.npy")
    matvecs = results["loss"][1]["num_matvecs"].sum()

    stats[labels[method]] = {
        "Test loss": results["loss"][0],
        "RMSE (Parameter)": results["rmse_param"],
        "Runtime/Epoch": jnp.mean(timestamps),
        "No. Matvecs": matvecs,
    }


stats = jax.tree_util.tree_map(float, stats)

results_dataframe = pd.DataFrame(stats)


# Create a latex-table
num_keys = len(stats["Arnoldi"].keys())
column_format = f"l{'c'*num_keys}"

latex = results_dataframe.to_latex(float_format="%.1e")
print()
print()
print(latex)
print()
print()
