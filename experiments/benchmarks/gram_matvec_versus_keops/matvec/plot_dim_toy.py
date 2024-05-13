import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

labels = {
    "gpytorch": "GPyTorch (via Keops)",
    "matfree_map": "Ours (via map)",
    "matfree_vmap_k10": "Ours (via vmap) K=10",
    "matfree_vmap_k100": "Ours (via vmap) K=100",
}
plt.rcParams.update(figsizes.neurips2021(ncols=3, nrows=2))
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.grid())
plt.rcParams.update(axes.legend())

directory = exp_util.matching_directory(__file__, "results/")

fig, axes = plt.subplots()

# Experiment set up
num_runs = 3
data_size = 100_000  # Number of samples
data_dim = [1, 3, 5, 10, 20, 30, 50, 80, 100, 120, 150, 200, 300, 500]
colors = ["#335c67", "#fff3b0", "#e09f3e", "#9e2a2b"]

for m, method in enumerate(
    ["gpytorch", "matfree_map", "matfree_vmap_k10", "matfree_vmap_k100"]
):
    times_min = []
    times_max = []
    times_mean = []
    for dim in data_dim:
        title = f"matvec_toy_num_runs_{num_runs}_data_dim_{dim}"
        x = jnp.load(f"{directory}{title}_{method}.npy")
        times_min.append(jnp.min(x))
        times_max.append(jnp.max(x))
        times_mean.append(jnp.mean(x))

        # times = jnp.asarray(times)

    axes.loglog(
        jnp.asarray(data_dim),
        jnp.asarray(times_min),
        color=colors[m],
        marker="^",
        markersize=5,
        label=labels[method],
    )


axes.set_title("Matvec run times (sec) vs. dim. of input data")
axes.set_xlabel("Data dim.")
axes.legend(fontsize="xx-small")
axes.set_ylabel("Run time (sec)")
axes.grid(which="major")

print("Saving to a file")
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)
plt.savefig(f"{directory}/figure_matvec_dim_toy.pdf")
