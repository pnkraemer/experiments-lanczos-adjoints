"""Create a beatiful figure for the paper."""

import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update(axes.grid())
plt.rcParams.update(fontsizes.iclr2024())
plt.rcParams.update(figsizes.iclr2024(ncols=3))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
args = parser.parse_args()
print(args)


# Decide which methods to plot
labels = {
    "arnoldi": "Arnoldi",
    # "diffrax:euler+backsolve": "Euler",
    # "diffrax:heun+recursive_checkpoint": "Heun",
    "diffrax:dopri5+backsolve": "Dopri5",
    "diffrax:tsit5+recursive_checkpoint": "Tsit5",
}
methods = list(labels.keys())

# Load the data set (here, this means loading the true parmater)
path = f"./data/pde_wave/{args.resolution}x{args.resolution}"
parameter = jnp.load(f"{path}_data_parameter.npy")
directory = exp_util.matching_directory(__file__, "results/")


# Prepare the figure: create subfigures, a meshgrid, etc.
xs_1d = jnp.linspace(0, 1, endpoint=True, num=len(parameter))
mesh = jnp.stack(jnp.meshgrid(xs_1d, xs_1d))


layout = [["truth", "arnoldi"], ["dopri5", "tsit5"]]
fig, ax = plt.subplot_mosaic(layout, figsize=(4, 3))


ax["truth"].set_title("Truth", fontsize="small")
img = ax["truth"].contourf(*mesh, jnp.abs(parameter), cmap="Greys")
plt.colorbar(img, ax=ax["truth"])

ax["arnoldi"].set_title(r"Solver: $\it{Arnoldi}$", fontsize="small")
path = f"{directory}{args.resolution}x{args.resolution}_arnoldi_s2"
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = ax["arnoldi"].contourf(*mesh, jnp.abs(parameter_estimate), cmap="Blues")
plt.colorbar(img, ax=ax["arnoldi"])

ax["dopri5"].set_title("Solver: Dopri5", fontsize="small")
path = f"{directory}{args.resolution}x{args.resolution}_diffrax:dopri5+backsolve_s2"
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = ax["dopri5"].contourf(*mesh, jnp.abs(parameter_estimate), cmap="Greens")
plt.colorbar(img, ax=ax["dopri5"])

ax["tsit5"].set_title("Solver: Tsit5", fontsize="small")
path = f"{directory}{args.resolution}x{args.resolution}_diffrax:tsit5+recursive_checkpoint_s2"
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = ax["tsit5"].contourf(*mesh, jnp.abs(parameter_estimate), cmap="Greens")
plt.colorbar(img, ax=ax["tsit5"])

# Set all x- and ylims to the unit square
for method in ["tsit5", "arnoldi", "dopri5"]:
    ax[method].sharex(ax["truth"])
    ax[method].sharey(ax["truth"])

    ax[method].set_xlim((0.0, 1.0))
    ax[method].set_xticks((0.0, 0.5, 1.0))
    ax[method].set_ylim((0.0, 1.0))
    ax[method].set_yticks((0.0, 0.5, 1.0))


directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
plt.savefig(f"{directory_fig}contours.pdf")


plt.show()


layout = ["forward", "forward", "gradient", "gradient", "loss", "loss", "loss"]
fig, ax = plt.subplot_mosaic([layout], dpi=200, figsize=(8, 2), constrained_layout=True)


for method in methods:
    num_matvecs = jnp.load(f"{directory}/wp_{method}_Ns.npy")
    # ts_all = jnp.load(f"{directory}/wp_{method}_ts.npy")
    errors_fwd = jnp.load(f"{directory}/wp_{method}_errors_fwd.npy")
    errors_rev = jnp.load(f"{directory}/wp_{method}_errors_rev.npy")
    idx = num_matvecs >= 3
    if "arnoldi" in method:
        alpha, zorder = 0.99, 100
    else:
        alpha, zorder = 0.8, 0

    ax["forward"].loglog(
        num_matvecs[idx],
        errors_fwd[idx],
        label=labels[method],
        alpha=alpha,
        zorder=zorder,
    )
    ax["gradient"].loglog(
        num_matvecs[idx],
        errors_rev[idx],
        label=labels[method],
        alpha=alpha,
        zorder=zorder,
    )

ax["forward"].set_ylabel("RMSE (relative)")
ax["forward"].set_title("Forward error")
ax["forward"].legend(fontsize="xx-small")
ax["forward"].grid(which="major")
ax["forward"].set_xlabel("No. Matvecs")
ax["gradient"].set_xlabel("No. Matvecs")
# ax["gradient"].set_ylabel("RMSE (relative)")
ax["gradient"].set_title("Gradient error")
ax["gradient"].legend(fontsize="xx-small")
ax["gradient"].grid(which="major")
ax["forward"].sharex(ax["gradient"])
ax["forward"].sharey(ax["gradient"])


ax["loss"].set_title("Training loss")
ax["loss"].set_xlabel("Time (sec)")
ax["loss"].set_ylabel("Loss")
for color, method in enumerate(methods):
    for seed in [1, 2, 3]:
        path = f"{directory}{args.resolution}x{args.resolution}_{method}_s{seed}"
        parameter_estimate = jnp.load(f"{path}_parameter.npy")
        loss = jnp.load(f"{path}_convergence.npy")
        timestamps = jnp.load(f"{path}_timestamps.npy")
        if "arnoldi" in method:
            alpha, zorder = 0.9, 100
        else:
            alpha, zorder = 0.9, 0

        def _process(x):
            return jnp.mean(x.reshape((-1, 50)), axis=-1)

        ax["loss"].semilogy(
            _process(timestamps),
            _process(loss),
            label=f"{labels[method]} (3000 epochs)",
            alpha=alpha,
            zorder=zorder,
            color=f"C{color}",
        )

        # handles, labels = ax["loss"].get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax["loss"].legend(by_label.values(), by_label.keys())

        handles_, labels_ = ax["loss"].get_legend_handles_labels()
        by_label = dict(zip(labels_, handles_))
        ax["loss"].legend(by_label.values(), by_label.keys(), fontsize="x-small")
        # ax["loss"].set_ylim((0, 100))
        ax["loss"].grid(axis="y", which="both")


kwargs = dict(fontsize="small", fontweight="bold", loc="left")
ax["loss"].set_title("B.", **kwargs)
ax["forward"].set_title("A1.", **kwargs)
ax["gradient"].set_title("A2.", **kwargs)

plt.savefig(f"{directory}pde_training_curves.pdf")

plt.show()


#
#
# assert False
# # Prepare the figure: create subfigures, a meshgrid, etc.
# xs_1d = jnp.linspace(0, 1, endpoint=True, num=len(parameter))
# mesh = jnp.stack(jnp.meshgrid(xs_1d, xs_1d))
# layout = [
#     ["convergence", "convergence", "truth", "arnoldi", "workprec_fwd"],
#     ["convergence", "convergence", "dopri5", "tsit5", "workprec_rev"],
# ]
# layout = onp.asarray(layout)
# fig, axes =
# plt.subplot_mosaic(layout, dpi=200, figsize=(8, 3), constrained_layout=True)
#
# # Plot the app
# # Plot the work-precision diagrams
# axes["convergence"].set_title("Convergence", fontsize="small")
# axes["convergence"].set_xlabel("Time (sec)", fontsize="small")
# axes["convergence"].set_ylabel("Loss", fontsize="small")
# for color, method in enumerate(methods):
#     for seed in [1, 2, 3]:
#         path = f"{directory}{args.resolution}x{args.resolution}_{method}_s{seed}"
#         parameter_estimate = jnp.load(f"{path}_parameter.npy")
#         convergence = jnp.load(f"{path}_convergence.npy")
#         timestamps = jnp.load(f"{path}_timestamps.npy")
#         if "arnoldi" in method:
#             alpha, zorder = 0.9, 100
#         else:
#             alpha, zorder = 0.9, 0
#
#         def _process(x):
#             return jnp.mean(x.reshape((-1, 50)), axis=-1)
#
#         axes["convergence"].semilogy(
#             _process(timestamps),
#             _process(convergence),
#             label=f"{labels[method]} (3000 epochs)",
#             alpha=alpha,
#             zorder=zorder,
#             color=f"C{color}",
#         )
#
#         # handles, labels = axes["convergence"].get_legend_handles_labels()
#         # by_label = dict(zip(labels, handles))
#         # axes["convergence"].legend(by_label.values(), by_label.keys())
#
#         handles_, labels_ = axes["convergence"].get_legend_handles_labels()
#         by_label = dict(zip(labels_, handles_))
#         axes["convergence"].legend(
#             by_label.values(), by_label.keys(), fontsize="x-small"
#         )
#         # axes["convergence"].set_ylim((0, 100))
#         axes["convergence"].grid(axis="y", which="both")
# #
# for method in methods:
#     num_matvecs = jnp.load(f"{directory}/wp_{method}_Ns.npy")
#     # ts_all = jnp.load(f"{directory}/wp_{method}_ts.npy")
#     errors_fwd = jnp.load(f"{directory}/wp_{method}_errors_fwd.npy")
#     errors_rev = jnp.load(f"{directory}/wp_{method}_errors_rev.npy")
#     idx = num_matvecs >= 3
#     if "arnoldi" in method:
#         alpha, zorder = 0.99, 100
#     else:
#         alpha, zorder = 0.8, 0
#
#     axes["workprec_fwd"].loglog(
#         num_matvecs[idx],
#         errors_fwd[idx],
#         label=labels[method],
#         alpha=alpha,
#         zorder=zorder,
#     )
#     axes["workprec_rev"].loglog(
#         num_matvecs[idx],
#         errors_rev[idx],
#         label=labels[method],
#         alpha=alpha,
#         zorder=zorder,
#     )
#
#
# axes["workprec_fwd"].sharex(axes["workprec_rev"])
# axes["workprec_fwd"].set_ylabel("RMSE (relative)", fontsize="small")
# axes["workprec_fwd"].set_title("Forward error", fontsize="small")
# axes["workprec_fwd"].legend(fontsize="xx-small")
# axes["workprec_fwd"].grid(which="major")
#
# axes["workprec_rev"].set_xlabel("No. Matvecs", fontsize="small")
# axes["workprec_rev"].set_ylabel("RMSE (relative)", fontsize="small")
# axes["workprec_rev"].set_title("Gradient error", fontsize="small")
# axes["workprec_rev"].legend(fontsize="xx-small")
# axes["workprec_rev"].grid(which="major")
#
#
# kwargs = dict(fontsize="small", fontweight="bold", loc="left")
# axes["convergence"].set_title("A.", **kwargs)
# axes["truth"].set_title("B1.", **kwargs)
# axes["arnoldi"].set_title("B2.", **kwargs)
# axes["dopri5"].set_title("B3.", **kwargs)
# axes["tsit5"].set_title("B4.", **kwargs)
# axes["workprec_fwd"].set_title("C1.", **kwargs)
# axes["workprec_rev"].set_title("C2.", **kwargs)
