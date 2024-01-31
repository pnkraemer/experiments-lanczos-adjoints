import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from tueplots import axes, bundles

from matfree_extensions import exp_util

# Plotting-configuration
plt.rcParams.update(bundles.neurips2023(nrows=1, ncols=2, family="sans-serif"))
plt.rcParams.update(axes.lines())


# Load the results
matrix_which = "t3dl_e"
directory = exp_util.matching_directory(__file__, "data/")
krylov_depths = jnp.load(f"{directory}/{matrix_which}_krylov_depths.npy")
times_fwdpass = jnp.load(f"{directory}/{matrix_which}_times_fwdpass.npy")
times_custom = jnp.load(f"{directory}/{matrix_which}_times_custom.npy")
times_autodiff = jnp.load(f"{directory}/{matrix_which}_times_autodiff.npy")
norms_of_differences = jnp.load(f"{directory}/{matrix_which}_norms_of_differences.npy")

# Download the matrix itself (for plotting sparsity patterns)
M = exp_util.suite_sparse_load(matrix_which, path="./data/matrices/")

# Create the three-column plot
fig, axes = plt.subplot_mosaic([["spy", "linear", "log"]], dpi=200)
label_fwd = "Forward"
label_adjoint = "Adjoint"
label_autodiff = "Backprop"

# Plot the sparsity pattern of the test-matrix
axes["spy"].set_title(f"SuiteSparse Matrix ({matrix_which})")
exp_util.plt_spy_coo(axes["spy"], M, cmap="viridis")
axes["spy"].set_xticks(())
axes["spy"].set_yticks(())

# Plot the linear scale
axes["linear"].set_title("Linear scale")
axes["linear"].plot(
    krylov_depths,
    times_fwdpass,
    linestyle="dashed",
    color="black",
    label=label_fwd,
)
axes["linear"].plot(krylov_depths, times_custom, label=label_adjoint)
axes["linear"].plot(krylov_depths, times_autodiff, label=label_autodiff)

# Plot the log-scale
axes["log"].set_title("Logarithmic scale")
axes["log"].semilogy(
    krylov_depths,
    times_fwdpass,
    linestyle="dashed",
    color="black",
    label=label_fwd,
)
axes["log"].semilogy(krylov_depths, times_custom, label=label_adjoint)
axes["log"].semilogy(krylov_depths, times_autodiff, label=label_autodiff)

# Label the benchmark plots
for name in ["log", "linear"]:
    axes[name].set_ylabel("Wall time (sec)")
    axes[name].set_xlabel("Krylov-space depth")
    axes[name].legend()
    axes[name].set_ylabel("Wall time (sec)")
    axes[name].set_xlabel("Krylov-space depth")

# Save the figure
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)
plt.savefig(f"{directory}/figure.pdf", dpi=150)
plt.show()
