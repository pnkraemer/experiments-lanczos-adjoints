"""Create a beatiful figure for the paper."""

import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp  # noqa: ICN001
from matfree_extensions.util import exp_util
from tueplots import axes, fontsizes

# plt.rcParams.update(figsizes.neurips2024(nrows=2, ncols=8, height_to_width_ratio=1.))

plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update(axes.grid())
plt.rcParams.update(fontsizes.neurips2024(default_smaller=2))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
args = parser.parse_args()
print(args)


labels = {
    "arnoldi": "Arnoldi",
    "diffrax:euler+backsolve": "Euler",
    "diffrax:heun+recursive_checkpoint": "Heun",
    "diffrax:dopri5+backsolve": "Dopri5",
    "diffrax:tsit5+recursive_checkpoint": "Tsit5",
}
methods = list(labels.keys())


path = f"./data/pde_wave/{args.resolution}x{args.resolution}"
parameter = jnp.load(f"{path}_data_parameter.npy")
directory = exp_util.matching_directory(__file__, "results/")

xs_1d = jnp.linspace(0, 1, endpoint=True, num=len(parameter))
mesh = jnp.stack(jnp.meshgrid(xs_1d, xs_1d))

layout = [
    ["convergence", "truth", "arnoldi", "workprec_fwd"],
    ["convergence", "dopri5", "tsit5", "workprec_rev"],
]
layout = onp.asarray(layout)

fig, axes = plt.subplot_mosaic(layout, dpi=200, figsize=(8, 3), constrained_layout=True)


axes["truth"].set_title("Truth", fontsize="small")
img = axes["truth"].contourf(*mesh, jnp.abs(parameter), cmap="Greys")
plt.colorbar(img, ax=axes["truth"])

axes["arnoldi"].set_title(r"Solver: $\it{Arnoldi}$", fontsize="small")
path = f"{directory}{args.resolution}x{args.resolution}_arnoldi"
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = axes["arnoldi"].contourf(*mesh, jnp.abs(parameter_estimate), cmap="Blues")
plt.colorbar(img, ax=axes["arnoldi"])

axes["dopri5"].set_title("Solver: Dopri5", fontsize="small")
path = f"{directory}{args.resolution}x{args.resolution}_diffrax:dopri5+backsolve"
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = axes["dopri5"].contourf(*mesh, jnp.abs(parameter_estimate), cmap="Greens")
plt.colorbar(img, ax=axes["dopri5"])

axes["tsit5"].set_title("Solver: Tsit5", fontsize="small")
path = (
    f"{directory}{args.resolution}x{args.resolution}_diffrax:tsit5+recursive_checkpoint"
)
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = axes["tsit5"].contourf(*mesh, jnp.abs(parameter_estimate), cmap="Greens")
plt.colorbar(img, ax=axes["tsit5"])


for method in ["tsit5", "arnoldi", "dopri5"]:
    axes[method].sharex(axes["truth"])
    axes[method].sharey(axes["truth"])

    axes[method].set_xlim((0.0, 1.0))
    axes[method].set_xticks((0.0, 0.5, 1.0))
    axes[method].set_ylim((0.0, 1.0))
    axes[method].set_yticks((0.0, 0.5, 1.0))

axes["convergence"].set_title("Convergence", fontsize="small")
axes["convergence"].set_xlabel("Time (sec)", fontsize="small")
axes["convergence"].set_ylabel("Loss", fontsize="small")
for method in methods:
    path = f"{directory}{args.resolution}x{args.resolution}_{method}"
    parameter_estimate = jnp.load(f"{path}_parameter.npy")
    convergence = jnp.load(f"{path}_convergence.npy")
    timestamps = jnp.load(f"{path}_timestamps.npy")
    if "arnoldi" in method:
        alpha, zorder = 0.99, 100
    else:
        alpha, zorder = 0.8, 0

    axes["convergence"].semilogy(
        timestamps, convergence, label=labels[method], zorder=zorder, alpha=alpha
    )
    axes["convergence"].legend(fontsize="x-small")
    axes["convergence"].grid(axis="y", which="both")

for method in methods:
    num_matvecs = jnp.load(f"{directory}/wp_{method}_Ns.npy")
    # ts_all = jnp.load(f"{directory}/wp_{method}_ts.npy")
    errors_fwd = jnp.load(f"{directory}/wp_{method}_errors_fwd.npy")
    errors_rev = jnp.load(f"{directory}/wp_{method}_errors_rev.npy")
    idx = num_matvecs >= 1
    if "arnoldi" in method:
        alpha, zorder = 0.99, 100
    else:
        alpha, zorder = 0.8, 0

    axes["workprec_fwd"].loglog(
        num_matvecs[idx],
        errors_fwd[idx],
        label=labels[method],
        alpha=alpha,
        zorder=zorder,
    )
    axes["workprec_rev"].loglog(
        num_matvecs[idx],
        errors_rev[idx],
        label=labels[method],
        alpha=alpha,
        zorder=zorder,
    )


axes["workprec_fwd"].sharex(axes["workprec_rev"])
axes["workprec_fwd"].set_ylabel("RMSE (relative)", fontsize="small")
axes["workprec_fwd"].set_title("Forward pass error", fontsize="small")
axes["workprec_fwd"].legend(fontsize="xx-small")
axes["workprec_fwd"].grid(which="major")

axes["workprec_rev"].set_xlabel("No. Matvecs", fontsize="small")
axes["workprec_rev"].set_ylabel("RMSE (relative)", fontsize="small")
axes["workprec_rev"].set_title("Gradient error", fontsize="small")
axes["workprec_rev"].legend(fontsize="xx-small")
axes["workprec_rev"].grid(which="major")


kwargs = dict(fontsize="small", fontweight="bold", loc="left")
axes["convergence"].set_title("A.", **kwargs)
axes["truth"].set_title("B1.", **kwargs)
axes["arnoldi"].set_title("B2.", **kwargs)
axes["dopri5"].set_title("B3.", **kwargs)
axes["tsit5"].set_title("B4.", **kwargs)
axes["workprec_fwd"].set_title("C1.", **kwargs)
axes["workprec_rev"].set_title("C2.", **kwargs)

directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}lookatme.pdf")

plt.show()
