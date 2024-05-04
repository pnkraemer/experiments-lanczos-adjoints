import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

# todo: reduce all benchmarks to map, vmap, and keops
# todo: clean up and simplify the plotting script
# todo: simplify the experiment code and write a bash script to run it

labels = {
    "gpytorch": "GPyTorch (via Keops)",
    "matfree_map": "Ours (via map)",
    "matfree_vmap": "Ours (via vmap)",
}
plt.rcParams.update(figsizes.neurips2024(ncols=2))
plt.rcParams.update(fontsizes.neurips2024())
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.grid())
plt.rcParams.update(axes.legend())

directory = exp_util.matching_directory(__file__, "results/")

fig, axes = plt.subplot_mosaic([["per_size", "per_dim"]], sharex=False, sharey=True)


print("Plotting data varied by dataset size")
num_runs = 3  # match to the existing data
data_dim = 1  # match to the existing data
data_sizes = 2 ** jnp.arange(1, 6, dtype=int)  # match to the existing data
for method in ["gpytorch", "matfree_map", "matfree_vmap"]:
    times = []
    for size in data_sizes:
        title = f"matvec_num_runs_{num_runs}_data_size_{size}_data_dim_{data_dim}"
        x = jnp.load(f"{directory}{title}_{method}.npy")
        times.append(x)
    times = jnp.asarray(times)
    axes["per_size"].semilogy(data_sizes, jnp.amin(times, axis=1), label=labels[method])

axes["per_size"].legend(fontsize="xx-small")
axes["per_size"].set_title(f"Dataset dimension: {data_dim}")
axes["per_size"].set_xlabel("Dataset size")
axes["per_size"].set_ylabel("Run time (sec)")
axes["per_size"].grid(which="major")

print("Plotting data varied by dataset dim")
num_runs = 3  # match to the existing data
data_size = 64  # match to the existing data
data_dims = jnp.arange(1, 8, dtype=int)  # match to the existing data

for method in ["gpytorch", "matfree_map", "matfree_vmap"]:
    times = []
    for dim in data_dims:
        title = f"matvec_num_runs_{num_runs}_data_size_{data_size}_data_dim_{dim}"
        x = jnp.load(f"{directory}{title}_{method}.npy")
        times.append(x)
    times = jnp.asarray(times)
    axes["per_dim"].semilogy(data_dims, jnp.amin(times, axis=1), label=labels[method])

axes["per_dim"].legend(fontsize="xx-small")
axes["per_dim"].set_title(f"Dataset size: {data_size}")
axes["per_dim"].set_xlabel("Dataset dim")
axes["per_dim"].grid(which="major")

print("Saving to a file")
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)
plt.savefig(f"{directory}/figure_matvec.pdf")

plt.show()
