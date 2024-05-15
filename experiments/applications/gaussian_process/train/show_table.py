"""Plot the results."""

import jax
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
            path = f"{directory}{name}75_{dataset}_s{seed}"
            ts = jnp.load(f"{path}_loss_timestamps.npy")
            times.append(ts[-1] / len(ts))

            ls = jnp.load(f"{path}_loss_values.npy")
            losses.append(ls[-1])

            rmses.append(jnp.load(f"{path}_rmse.npy"))

        results[dataset][name]["times"] = jnp.stack(times, axis=-1)
        results[dataset][name]["rmses"] = jnp.stack(rmses, axis=-1)
        results[dataset][name]["losses"] = jnp.stack(losses, axis=-1)

    for name in ["adjoints"]:
        results[dataset][name] = {}
        times, rmses, losses = [], [], []
        for seed in [1, 2, 3]:
            path = f"{directory}{name}75_{dataset}_s{seed}"

            ts = jnp.load(f"{path}_loss_timestamps.npy")
            times.append(ts[-1] / len(ts))

            ls = jnp.load(f"{path}_loss_curve.npy")
            losses.append(ls[-1])

            rmses.append(jnp.load(f"{path}_test_rmses.npy"))

        results[dataset][name]["times"] = jnp.stack(times, axis=-1)
        results[dataset][name]["rmses"] = jnp.stack(rmses, axis=-1)
        results[dataset][name]["losses"] = jnp.stack(losses, axis=-1)


def stats(x):
    if len(x) > 50:
        x = x[-50:]
    return {
        "mean": f"{jnp.round(jnp.mean(x, axis=-1), 2):.2f}",
        "std": f"{jnp.round(jnp.std(x, axis=-1), 3):.3f}",
        # "min": jnp.amin(x, axis=-1),
        # "max": jnp.amax(x, axis=-1),
    }


data = jax.tree_util.tree_map(stats, results)
for d in datasets:
    print(d)
    data_ = data[d]
    for meth in ["adjoints", "gpytorch"]:
        print(meth)
        res = data_[meth]
        for x in ["rmses", "losses", "times"]:
            print(x)
            print(res[x])

        print()
    print()
#
# for dataset, content in data.items():
#     print(dataset)
#     for method, results in content.items():
#         print(method)
#         for a, b in results.items():
#             print(a)
#             print(b)
#
#         print()
#     print()
