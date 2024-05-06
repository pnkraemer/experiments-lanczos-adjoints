import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tueplots import axes, figsizes, fontsizes

# Plotting-configuration
plt.rcParams.update(
    figsizes.neurips2024(nrows=1, ncols=1, rel_width=0.4, height_to_width_ratio=0.8)
)
plt.rcParams.update(fontsizes.neurips2024(default_smaller=2))
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.spines(top=False, right=False, bottom=True, left=True))


# Load the results
matrix_which = "t3dl_e"
directory = exp_util.matching_directory(__file__, "results/")
krylov_depths = jnp.load(f"{directory}{matrix_which}_krylov_depths.npy")
times_fwdpass = jnp.load(f"{directory}{matrix_which}_times_fwdpass.npy")
times_custom = jnp.load(f"{directory}{matrix_which}_times_custom.npy")
times_autodiff = jnp.load(f"{directory}{matrix_which}_times_autodiff.npy")
norms_of_differences = jnp.load(f"{directory}{matrix_which}_norms_of_differences.npy")

# Download the matrix itself (for plotting sparsity patterns)
M = exp_util.suite_sparse_load(matrix_which, path="./data/matrices/")

# Create the three-column plot
fig, ax = plt.subplots(dpi=200)


style_fwd = {"label": "Forward", "color": "black", "linestyle": "dashed"}
ax.plot(krylov_depths, times_fwdpass, **style_fwd)
ax.annotate("Forward pass", xy=(60.0, -0.005), color="black", fontsize="xx-small")

style_adjoint = {"label": "Adjoint", "color": "C0"}
ax.plot(krylov_depths, times_custom, **style_adjoint)
ax.annotate("Adjoint", xy=(70.0, 0.04), color="C0", fontsize="x-small")

style_autodiff = {"label": "Backprop", "color": "C1"}
ax.plot(krylov_depths, times_autodiff, **style_autodiff)
ax.annotate("Backprop", xy=(60.0, 0.2), color="C1", fontsize="x-small")


with plt.rc_context(axes.spines(top=True, right=True, bottom=True, left=True)):
    ax_in = inset_axes(ax, width="30%", height="30%", loc=2, borderpad=1.5)
    ax_in.set_title(f"SuiteSparse: {matrix_which}", fontsize="xx-small", pad=3)
    exp_util.plt_spy_coo(ax_in, M, markersize=2, cmap="viridis", invert_axes=False)
    ax_in.set_xlabel(rf"$N={len(M.data)}$ rows", fontsize="xx-small")
    ax_in.invert_yaxis()
    ax_in.set_xticks(())
    ax_in.set_yticks(())


ax.set_xlabel("Krylov-space depth", fontsize="small")
ax.set_ylabel("Wall time (sec)", fontsize="small")

ax.set_xlim((-1, 90))


# # Save the figure
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)
plt.savefig(f"{directory}/figure_single.pdf")
plt.show()