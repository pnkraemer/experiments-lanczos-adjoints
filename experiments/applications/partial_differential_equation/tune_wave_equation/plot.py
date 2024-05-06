import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp  # for matplotlib manipulations  # noqa: ICN001
from matfree_extensions.util import exp_util

# todo: plot all methods next to each other

directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")

parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
args = parser.parse_args()


y0 = jnp.load(f"{directory_results}{args.method}_y0.npy")
scale_mlp_before = jnp.load(f"{directory_results}{args.method}_scale_mlp_before.npy")
scale_mlp_after = jnp.load(f"{directory_results}{args.method}_scale_mlp_after.npy")
scale_grf = jnp.load(f"{directory_results}{args.method}_scale_grf.npy")
y1_target = jnp.load(f"{directory_results}{args.method}_y1_target.npy")
y1_approx_before = jnp.load(f"{directory_results}{args.method}_y1_approx_before.npy")
y1_approx_after = jnp.load(f"{directory_results}{args.method}_y1_approx_after.npy")


# Plot

layout = onp.asarray(
    [
        ["truth_scale", "truth_t0", "truth_t1"],
        ["before_scale", "before_t0", "before_t1"],
        ["after_scale", "after_t0", "after_t1"],
    ]
)
figsize = (onp.shape(layout)[1] * 3, onp.shape(layout)[0] * 2)
fig, axes = plt.subplot_mosaic(layout, figsize=figsize, sharex=True, sharey=True)


def plot_t0(ax, x, /):
    kwargs_t0 = {"cmap": "Greys"}
    args_plot = x

    clr = ax.contourf(args_plot, **kwargs_t0)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_t1(ax, x, /):
    kwargs_t1 = {"cmap": "Oranges"}
    args_plot = x

    clr = ax.contourf(args_plot, **kwargs_t1)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_scale(ax, x, /):
    kwargs_scale = {"cmap": "Blues"}
    args_plot = x
    clr = ax.contourf(args_plot, **kwargs_scale)
    fig.colorbar(clr, ax=ax)
    return ax


axes["truth_t0"].set_title("$y(t=t_0)$ (known)", fontsize="medium")
axes["truth_t1"].set_title("$y(t=t_1)$ (target)", fontsize="medium")
axes["truth_scale"].set_title("GRF / MLP (unknown)", fontsize="medium")

axes["truth_scale"].set_ylabel("Truth (GRF)")
plot_t0(axes["truth_t0"], y0[0])
plot_t1(axes["truth_t1"], y1_target[0])
plot_scale(axes["truth_scale"], scale_grf)


axes["before_scale"].set_ylabel("Before optim. (MLP)")
plot_t0(axes["before_t0"], y0[0])
plot_t1(axes["before_t1"], y1_approx_before[0])
plot_scale(axes["before_scale"], scale_mlp_before)


axes["after_scale"].set_ylabel("After optim. (MLP)")
plot_t0(axes["after_t0"], y0[0])
plot_t1(axes["after_t1"], y1_approx_after[0])
plot_scale(axes["after_scale"], scale_mlp_after)

plt.savefig(f"{directory_fig}/figure.pdf")
plt.show()
