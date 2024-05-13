import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

labels = {
    "gpytorch": "GPyTorch (via Keops)",
    "matfree_full": "Ours",
    "matfree_sequential": "Ours (via map)",
    "matfree_partitioned_10": "Ours (via vmap) K=10",
    "matfree_partitioned_100": "Ours (via vmap) K=100",
    "matfree_partitioned_1000": "Ours (via vmap) K=1000",
}
plt.rcParams.update(figsizes.neurips2021(ncols=3, nrows=2))
plt.rcParams.update(fontsizes.neurips2021())
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.grid())
plt.rcParams.update(axes.legend())

directory = exp_util.matching_directory(__file__, "results/")

fig, axes = plt.subplots()

# Experiment set up
num_runs = 5
data_dim = 30  # Number of samples
data_size = [
    100,
    300,
    500,
    1_000,
    2_000,
    3_000,
    5_000,
    8_000,
    10_000,
    12_000,
    15_000,
    20_000,
    30_000,
    50_000,
    100_000,
    200_000,
    300_000,
    500_000,
]
# colors = ['#335c67','#fff3b0','#e09f3e', '#9e2a2b', '#540b0e', '#bce784']
colors = ["#335c67", "#5e60ce", "#e09f3e", "#9e2a2b", "#540b0e", "#ea7317"]

for m, method in enumerate(
    [
        "gpytorch",
        "matfree_full",
        "matfree_sequential",
        "matfree_partitioned_10",
        "matfree_partitioned_100",
        "matfree_partitioned_1000",
    ]
):
    times_min = []
    times_max = []
    times_mean = []
    for size in data_size:
        title = f"matvec_toy_num_runs_{num_runs}_data_size_{size}"
        try:
            x = jnp.load(f"{directory}{title}_{method}.npy")
            times_min.append(jnp.min(x))
            times_max.append(jnp.max(x))
            times_mean.append(jnp.mean(x))
        except Exception as e:
            print(e)

    if len(data_size) > len(times_min):
        data_size_plot = data_size[: len(times_min)]
        axes.loglog(
            jnp.asarray(data_size_plot),
            jnp.asarray(times_min),
            color=colors[m],
            marker="^",
            markersize=3,
            label=labels[method],
        )
    else:
        axes.loglog(
            jnp.asarray(data_size),
            jnp.asarray(times_min),
            color=colors[m],
            marker="^",
            markersize=3,
            label=labels[method],
        )


axes.set_title("Matvec run times (sec) vs. Dataset size")
axes.set_xlabel("Dataset size")
axes.legend(fontsize="xx-small")
axes.set_ylabel("Run time (sec)")
axes.grid(which="major")

print("Saving to a file")
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)
plt.savefig(f"{directory}/figure_matvec_size_toy.pdf")
