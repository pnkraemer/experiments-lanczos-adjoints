"""Plot the results."""

import jax
import jax.numpy as jnp
from matfree_extensions.util import exp_util

datasets = [
    "concrete",
    "power_plant",
    "elevators",
    "protein",
    "kin40k",
    "kegg_directed",
    "kegg_undirected",
]
directory = exp_util.matching_directory(__file__, "results/")


results: dict = {}

for name in ["adjoints"]:
    results[name] = {}
    for dataset in datasets:
        results[name][dataset] = {}
        times, rmses, losses = [], [], []
        for seed in [1, 2, 3, 4, 5]:
            path = f"{directory}final-{name}_{dataset}_s{seed}"

            times.append(jnp.load(f"{path}_loss_timestamps.npy"))
            rmses.append(jnp.load(f"{path}_test_rmses.npy"))
            losses.append(jnp.load(f"{path}_loss_curve.npy"))

        results[name][dataset]["times"] = jnp.stack(times, axis=-1)
        results[name][dataset]["rmses"] = jnp.stack(rmses, axis=-1)
        results[name][dataset]["losses"] = jnp.stack(losses, axis=-1)

for name in ["gpytorch"]:
    results[name] = {}
    for dataset in datasets:
        results[name][dataset] = {}
        times, rmses, losses = [], [], []
        for seed in [1, 2, 3, 4, 5]:
            path = f"{directory}final-{name}_{dataset}_s{seed}"
            times.append(jnp.load(f"{path}_loss_timestamps.npy"))
            rmses.append(jnp.load(f"{path}_rmse.npy"))
            losses.append(jnp.load(f"{path}_loss_values.npy"))

        results[name][dataset]["times"] = jnp.stack(times, axis=-1)
        results[name][dataset]["rmses"] = jnp.stack(rmses, axis=-1)
        results[name][dataset]["losses"] = jnp.stack(losses, axis=-1)


print(jax.tree_util.tree_map(jnp.shape, results))
