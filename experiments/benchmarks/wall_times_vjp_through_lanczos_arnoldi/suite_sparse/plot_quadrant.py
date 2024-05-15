import itertools
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

# Plotting-configuration
plt.rcParams.update(
    figsizes.neurips2024(nrows=1, ncols=1, rel_width=0.4, height_to_width_ratio=0.8)
)
plt.rcParams.update(fontsizes.neurips2024(default_smaller=2))
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.spines(top=False, right=False, bottom=True, left=True))


# Load the results
matrix = "bcsstk18"
reortho = "none"
backprop_until = 100
max_krylov_depth = 150

# precompile = True
# method = "lanczos"


# Create the three-column plot
# Download the matrix itself (for plotting sparsity patterns)
M = exp_util.suite_sparse_load(matrix, path="./data/matrices/")
directory = exp_util.matching_directory(__file__, "results/")


fig, axes_ = plt.subplots(dpi=200, ncols=4, figsize=(8, 1.5))
for ax, (method, precompile) in zip(
    axes_, itertools.product(["lanczos", "arnoldi"], [True, False])
):
    print(method, precompile)

    num_runs = 5 if precompile else 1
    path = (
        f"{directory}{method}_{matrix}"
        f"_reortho_{reortho}"
        f"_num_runs_{num_runs}"
        f"_backprop_until_{backprop_until}"
        f"_max_krylov_depth_{max_krylov_depth}"
        f"_precompile_{precompile}"
    )
    krylov_depths = jnp.load(f"{path}_krylov_depths.npy")
    times_fwdpass = jnp.load(f"{path}_times_fwdpass.npy")
    times_custom = jnp.load(f"{path}_times_custom.npy")
    times_autodiff = jnp.load(f"{path}_times_autodiff.npy")

    title = f"{method.title()}: {'Run' if precompile else 'Compile'} time"
    ax.set_title(title, fontsize="medium")
    style_fwd = {"label": "Forward", "color": "black", "linestyle": "dashed"}
    ax.plot(krylov_depths, times_fwdpass, **style_fwd)
    # ax.annotate("Forward pass", xy=(60.0, -0.005), color="black", fontsize="xx-small")

    style_adjoint = {"label": "Adjoint", "color": "C0"}
    ax.plot(krylov_depths, times_custom, **style_adjoint)
    # ax.annotate("Adjoint", xy=(70.0, 0.04), color="C0", fontsize="x-small")

    style_autodiff = {"label": "Backprop", "color": "C1"}
    ax.plot(krylov_depths[: len(times_autodiff)], times_autodiff, **style_autodiff)
    # ax.annotate("Backprop", xy=(60.0, 0.2), color="C1", fontsize="x-small")

    ax.legend(fontsize="xx-small")
    ax.set_xlabel("Krylov-space depth", fontsize="small")
    ax.set_ylabel("Wall time (sec)", fontsize="small")

    # ax.set_xlim((-1, 90))
    # ax.set_ylim((0., 0.5))

# # Save the figure
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)
# plt.savefig(f"{directory}/figure_single.pdf")
plt.show()
