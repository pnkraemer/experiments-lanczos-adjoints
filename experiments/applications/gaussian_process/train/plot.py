"""Plot the results."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from tueplots import axes

datasets = [
    # "concrete",
    # "power_plant",
    "elevators",
    "protein",
    "kin40k",
    "kegg_directed",
    "kegg_undirected",
]
labelmap = {"gpytorch": "GPyTorch", "adjoints": "Ours"}

# plt.rcParams.update(bundles.neurips2024(ncols=len(datasets), family="sans-serif"))
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())

directory = exp_util.matching_directory(__file__, "results/")


results: dict = {}

for dataset in datasets:
    results[dataset] = {}
    for name in ["gpytorch"]:
        results[dataset][name] = {}
        times, rmses, losses = [], [], []
        for seed in [1, 2, 3]:
            path = f"{directory}final-{name}_{dataset}_s{seed}"
            times.append(jnp.load(f"{path}_loss_timestamps.npy"))
            rmses.append(jnp.load(f"{path}_rmse.npy"))
            losses.append(jnp.load(f"{path}_loss_values.npy"))

        results[dataset][name]["times"] = jnp.stack(times, axis=-1)
        results[dataset][name]["rmses"] = jnp.stack(rmses, axis=-1)
        results[dataset][name]["losses"] = jnp.stack(losses, axis=-1)

    for name in ["adjoints"]:
        results[dataset][name] = {}
        times, rmses, losses = [], [], []
        for seed in [1, 2, 3]:
            path = f"{directory}final-{name}_{dataset}_s{seed}"

            times.append(jnp.load(f"{path}_loss_timestamps.npy"))
            rmses.append(jnp.load(f"{path}_test_rmses.npy"))
            losses.append(jnp.load(f"{path}_loss_curve.npy"))

        results[dataset][name]["times"] = jnp.stack(times, axis=-1)
        results[dataset][name]["rmses"] = jnp.stack(rmses, axis=-1)
        results[dataset][name]["losses"] = jnp.stack(losses, axis=-1)


def stats(x):
    if len(x) > 50:
        x = x[-50:]
    return {
        "mean": jnp.mean(x, axis=-1),
        "std": jnp.std(x, axis=-1),
        "min": jnp.amin(x, axis=-1),
        "max": jnp.amax(x, axis=-1),
    }


# results_stats = jax.tree_util.tree_map(stats, results)
figsize = (8, 2)
fig, ax = plt.subplot_mosaic(
    [datasets], figsize=figsize, dpi=200, constrained_layout=True
)
for data, data_results in results.items():
    ax[data].set_title(data, fontsize="medium")

    for color, (name, values) in enumerate(data_results.items()):
        ax[data].plot(values["losses"], color=f"C{color}", label=labelmap[name])

    handles, labels = ax[data].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[data].legend(by_label.values(), by_label.keys(), fontsize="x-small")

    ax[data].set_xlabel("Epoch", fontsize="medium")

ax[datasets[0]].set_ylabel("Loss", fontsize="medium")

directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
plt.savefig(f"{directory_fig}loss_curves.pdf")
plt.show()

print()
