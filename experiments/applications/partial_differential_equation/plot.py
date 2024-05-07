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


layout = [
    ["convergence", "truth", "arnoldi", "workprec"],
    ["convergence", "dopri5", "tsit5", "workprec"],
]
layout = onp.asarray(layout)

fig, axes = plt.subplot_mosaic(layout, dpi=200, figsize=(8, 2), constrained_layout=True)


axes["truth"].set_title("Truth", fontsize="small")
img = axes["truth"].contourf(jnp.abs(parameter))
plt.colorbar(img, ax=axes["truth"])

axes["arnoldi"].set_title("Arnoldi", fontsize="small")
path = f"{directory}{args.resolution}x{args.resolution}_arnoldi"
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = axes["arnoldi"].contourf(jnp.abs(parameter_estimate))
plt.colorbar(img, ax=axes["arnoldi"])

axes["dopri5"].set_title("Dopri5", fontsize="small")
path = f"{directory}{args.resolution}x{args.resolution}_diffrax:dopri5+backsolve"
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = axes["dopri5"].contourf(jnp.abs(parameter_estimate))
plt.colorbar(img, ax=axes["dopri5"])

axes["tsit5"].set_title("Tsit5", fontsize="small")
path = (
    f"{directory}{args.resolution}x{args.resolution}_diffrax:tsit5+recursive_checkpoint"
)
parameter_estimate = jnp.load(f"{path}_parameter.npy")
img = axes["tsit5"].contourf(jnp.abs(parameter_estimate))
plt.colorbar(img, ax=axes["tsit5"])


axes["convergence"].set_title("Convergence", fontsize="small")
axes["convergence"].set_xlabel("Time (sec)", fontsize="small")
axes["convergence"].set_ylabel("Loss", fontsize="small")
for method in methods:
    path = f"{directory}{args.resolution}x{args.resolution}_{method}"
    parameter_estimate = jnp.load(f"{path}_parameter.npy")
    convergence = jnp.load(f"{path}_convergence.npy")
    timestamps = jnp.load(f"{path}_timestamps.npy")
    axes["convergence"].semilogy(timestamps, convergence, label=labels[method])
axes["convergence"].legend(fontsize="xx-small")

for method in methods:
    num_matvecs = jnp.load(f"{directory}/wp_{method}_Ns.npy")
    # ts_all = jnp.load(f"{directory}/wp_{method}_ts.npy")
    # errors_fwd = jnp.load(f"{directory}/wp_{method}_errors_fwd.npy")
    errors_rev = jnp.load(f"{directory}/wp_{method}_errors_rev.npy")
    idx = num_matvecs >= 1
    axes["workprec"].loglog(num_matvecs[idx], errors_rev[idx], label=labels[method])

axes["workprec"].set_xlabel("No. Matvecs", fontsize="small")
axes["workprec"].set_ylabel("RMSE (relative)", fontsize="small")
axes["workprec"].set_title("Gradient error", fontsize="small")
axes["workprec"].legend(fontsize="xx-small")
axes["workprec"].grid(which="major")

directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}lookatme.pdf")

plt.show()
