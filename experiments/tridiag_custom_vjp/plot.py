import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from matfree_extensions import exp_util

# from tueplots import axes, bundles, markers


directory = exp_util.matching_directory(__file__, "data/")

matrix_which = "t3dl_e"
krylov_depths = jnp.load(f"{directory}/{matrix_which}_krylov_depths.npy")
times_fwdpass = jnp.load(f"{directory}/{matrix_which}_times_fwdpass.npy")
times_custom = jnp.load(f"{directory}/{matrix_which}_times_custom.npy")
times_autodiff = jnp.load(f"{directory}/{matrix_which}_times_autodiff.npy")
norms_of_differences = jnp.load(f"{directory}/{matrix_which}_norms_of_differences.npy")

fig, axes = plt.subplot_mosaic(
    [["linear"]], sharex=True, figsize=(5, 3), constrained_layout=True
)

# axes["linear"].set_title("Linear scale")
axes["linear"].plot(
    krylov_depths,
    times_fwdpass,
    linestyle="dashed",
    color="black",
    label="Forward-pass",
)
axes["linear"].plot(krylov_depths, times_custom, label="Lanczos-adjoint")
axes["linear"].plot(krylov_depths, times_autodiff, label="Auto-diff")
# axes["linear"].plot(krylov_depths, norms_of_differences, label="Forward-pass")
axes["linear"].legend()
#
axes["linear"].set_title(f"SuiteSparse: {matrix_which}")

axes["linear"].set_ylabel("Wall time (sec)")
axes["linear"].set_xlabel("Krylov-space depth")
# axes["log"].set_title("Logarithmic scale")
# axes["log"].semilogy(krylov_depths, times_fwdpass)
# axes["log"].semilogy(krylov_depths, times_custom)
# axes["log"].semilogy(krylov_depths, times_autodiff)
# axes["log"].semilogy(krylov_depths, norms_of_differences)


directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)
plt.savefig(f"{directory}/figure.pdf", dpi=150)
plt.show()
