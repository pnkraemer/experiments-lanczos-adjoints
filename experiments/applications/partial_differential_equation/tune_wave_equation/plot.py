import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp  # for matplotlib manipulations  # noqa: ICN001
import pandas as pd
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

# todo: plot all methods next to each other
# todo: get all stats into a datafram and print latex

directory_results = exp_util.matching_directory(__file__, "results/")
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)

labels = {
    # "expm-pade": "Naive (Pade)",
    # "euler": "Euler",
    "diffrax-euler": "Euler (Diffrax)",
    "diffrax-tsit5": "Tsit5 (Diffrax)",
    "arnoldi": "Arnoldi",
}
stats = {}
for method, label in labels.items():
    with open(f"{directory_results}{method}_stats.pkl", "rb") as handle:
        stats[label] = pickle.load(handle)


stats_frame = pd.DataFrame(stats).T

num_stats = len(stats["Arnoldi"].keys())
column_format = f"l{'c'*num_stats}"
latex = stats_frame.to_latex(column_format=column_format, float_format="%.1e")

print("\n")
print("\n")
print(latex)
print("\n")
print("\n")


# method = "arnoldi"
# y0 = jnp.load(f"{directory_results}{method}_y0.npy")
# scale_mlp_before = jnp.load(f"{directory_results}{method}_scale_mlp_before.npy")
# scale_mlp_after = jnp.load(f"{directory_results}{method}_scale_mlp_after.npy")
# scale_grf = jnp.load(f"{directory_results}{method}_scale_grf.npy")
# y1_target = jnp.load(f"{directory_results}{method}_y1_target.npy")
# y1_approx_before = jnp.load(f"{directory_results}{method}_y1_approx_before.npy")
# y1_approx_after = jnp.load(f"{directory_results}{method}_y1_approx_after.npy")


# Plot


def plot_t0(ax, z, /):
    kwargs_t0 = {"cmap": "Greys"}
    x = jnp.linspace(0, 1, endpoint=True, num=z.shape[0])
    y = jnp.linspace(0, 1, endpoint=True, num=z.shape[1])
    x0, x1 = jnp.meshgrid(x, y)
    clr = ax.pcolormesh(x0, x1, z, **kwargs_t0)
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.tick_params(axis="both", which="major", labelsize="xx-small")
    cbar = fig.colorbar(clr, ax=ax)
    cbar.ax.tick_params(labelsize="xx-small")
    return ax


def plot_t1(ax, z, /):
    kwargs_t1 = {"cmap": "Oranges"}
    x = jnp.linspace(0, 1, endpoint=True, num=z.shape[0])
    y = jnp.linspace(0, 1, endpoint=True, num=z.shape[1])
    x0, x1 = jnp.meshgrid(x, y)
    clr = ax.pcolormesh(x0, x1, z, **kwargs_t1)
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.tick_params(axis="both", which="major", labelsize="xx-small")

    cbar = fig.colorbar(clr, ax=ax)
    cbar.ax.tick_params(labelsize="xx-small")
    return ax


def plot_scale(ax, z, /):
    kwargs_scale = {"cmap": "Blues"}
    x = jnp.linspace(0, 1, endpoint=True, num=z.shape[0])
    y = jnp.linspace(0, 1, endpoint=True, num=z.shape[1])
    x0, x1 = jnp.meshgrid(x, y)
    clr = ax.pcolormesh(x0, x1, z, **kwargs_scale)
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.tick_params(axis="both", which="major", labelsize="xx-small")

    cbar = fig.colorbar(clr, ax=ax)
    cbar.ax.tick_params(labelsize="xx-small")
    return ax


label_col = ["truth", *list(labels.keys())]
label_row = ["param", "y0", "y1"]

layout = [[f"{what}_{how}" for how in label_col] for what in label_row]
layout = onp.asarray(layout)

nrows, ncols = onp.shape(layout)
plt.rcParams.update(figsizes.neurips2024(nrows=nrows, ncols=ncols))

plt.rcParams.update(fontsizes.neurips2024(default_smaller=2))
plt.rcParams.update(axes.lines())

fig, axes = plt.subplot_mosaic(layout, sharex=True, sharey=True, dpi=200)

print("Plotting the truth")
y0 = jnp.load(f"{directory_results}arnoldi_y0.npy")
scale_grf = jnp.load(f"{directory_results}arnoldi_scale_grf.npy")
y1_target = jnp.load(f"{directory_results}arnoldi_y1_target.npy")

axes["param_truth"].set_title("Truth", fontsize="medium")
plot_scale(axes["param_truth"], scale_grf)
plot_t0(axes["y0_truth"], y0[0])
plot_t1(axes["y1_truth"], y1_target[0])


axes["param_truth"].set_ylabel("Parameter", fontsize="small")
axes["y0_truth"].set_ylabel("Known: $y(t_0)$", fontsize="small")
axes["y1_truth"].set_ylabel("Target: $y(t_1)$", fontsize="small")

for method, label in labels.items():
    y0 = jnp.load(f"{directory_results}{method}_y0.npy")
    scale_mlp_after = jnp.load(f"{directory_results}{method}_scale_mlp_after.npy")
    y1_approx_after = jnp.load(f"{directory_results}{method}_y1_approx_after.npy")

    axes[f"param_{method}"].set_title(label, fontsize="small")
    plot_scale(axes[f"param_{method}"], scale_grf)
    plot_t0(axes[f"y0_{method}"], y0[0])
    plot_t1(axes[f"y1_{method}"], y1_target[0])


fig.align_ylabels()

plt.savefig(f"{directory_fig}/figure.pdf")
plt.show()
