"""Create a beatiful figure for the paper."""

import argparse
import os 

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp  # noqa: ICN001
from matfree_extensions.util import exp_util
from tueplots import axes, fontsizes

plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update(axes.grid())
plt.rcParams.update(fontsizes.neurips2024())

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
args = parser.parse_args()
print(args)


labels = {
    "arnoldi": "Arnoldi",
    # "diffrax:euler+backsolve": "Euler/Backsolve (Diffrax)",
    # "diffrax:heun+recursive_checkpoint": "Heun/AD (Diffrax)",
    "diffrax:dopri5+backsolve": "Dopri5/Backsolve (Diffrax)",
    "diffrax:tsit5+recursive_checkpoint": "Tsit5/AD (Diffrax)",
}
methods = list(labels.keys())


path = f"./data/pde_wave/{args.resolution}x{args.resolution}"
parameter = jnp.load(f"{path}_data_parameter.npy")
directory = exp_util.matching_directory(__file__, "results/")

layout = onp.asarray([["truth", *methods, "convergence"]])
figsize = ((len(methods) + 2) * 2, 2)
# plt.rcParams.update(figsizes.neurips2024(nrows=len(layout), ncols=len(layout.T)))
fig, axes = plt.subplot_mosaic(
    layout, figsize=figsize, dpi=100, constrained_layout=True
)

axes["truth"].set_title("Truth", fontsize="medium")
img = axes["truth"].contourf(jnp.abs(parameter))
plt.colorbar(img, ax=axes["truth"])

for method in methods:
    path = f"{directory}{args.resolution}x{args.resolution}_{method}"
    parameter_estimate = jnp.load(f"{path}_parameter.npy")
    convergence = jnp.load(f"{path}_convergence.npy")
    timestamps = jnp.load(f"{path}_timestamps.npy")

    axes[method].set_title(labels[method], fontsize="medium")
    img = axes[method].contourf(jnp.abs(parameter_estimate))
    plt.colorbar(img, ax=axes[method])

    axes["convergence"].semilogy(timestamps, convergence, label=labels[method])

axes["convergence"].set_title("Convergence", fontsize="medium")
axes["convergence"].set_xlabel("Time (sec)", fontsize="medium")
axes["convergence"].set_ylabel("Loss", fontsize="medium")
plt.legend()


directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}lookatme.pdf")

plt.show()
