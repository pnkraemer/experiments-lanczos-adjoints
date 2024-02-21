import functools
import os

import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
from matfree import test_util
from matfree_extensions import arnoldi, exp_util
from tueplots import axes, figsizes, fontsizes

# Set a few display-related parameters
plt.rcParams.update(figsizes.icml2024_full(ncols=4, nrows=1, height_to_width_ratio=1.0))
plt.rcParams.update(fontsizes.icml2024())
plt.rcParams.update(axes.lines())
jnp.set_printoptions(1, suppress=False)


# A few auxiliary functions


def magnitude(val, /):
    epsilon = jnp.finfo(jnp.dtype(val)).eps
    return jnp.log10(epsilon + jnp.abs(val))


def matvec(x, p):
    return p @ x


nrows = 21

fig = plt.figure(layout="constrained", dpi=200)
fig_single, fig_double = fig.subfigures(nrows=1, ncols=2)


for x64, subfig, title, cmap in zip(
    [False, True],
    [fig_single, fig_double],
    ["Single precision", "Double precision"],
    ["copper", "bone"],
):
    jax.config.update("jax_enable_x64", x64)

    axes = subfig.subplot_mosaic([[*(["none"] * 10), "arrow", *(["full"] * 10)]])
    axes["none"].set_title("No proj.")
    axes["full"].set_title("Full proj.")

    arrowprops = {"arrowstyle": "->"}
    axes["arrow"].annotate("", xy=(0.0, 1.0), xytext=(0, 0), arrowprops=arrowprops)
    axes["arrow"].set_xticks(())
    axes["arrow"].set_yticks(())
    axes["arrow"].spines["top"].set_visible(False)
    axes["arrow"].spines["bottom"].set_visible(False)
    axes["arrow"].spines["left"].set_visible(False)
    axes["arrow"].spines["right"].set_visible(False)

    for key in ["none", "full"]:
        axes[key].set_xlabel("Column index", fontsize="small")
        axes[key].set_xticks((0, (nrows - 1) // 2, nrows - 1))
        axes[key].set_xticklabels((1, nrows // 2, nrows))

        axes[key].set_ylabel("Row index", fontsize="small")
        axes[key].set_yticks((0, (nrows - 1) // 2, nrows - 1))
        axes[key].set_yticklabels((1, nrows // 2, nrows))

    for reortho in ["none", "full"]:
        fwd = functools.partial(arnoldi.forward, reortho="full")
        adj = functools.partial(arnoldi.adjoint, reortho=reortho)

        # Set up a test problem
        eigvals = 2.0 ** jnp.arange(-nrows // 2, nrows // 2, 1)
        matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
        vector = jnp.ones((nrows,)) / jnp.sqrt(nrows)

        # Forward pass
        krylov_depth = len(vector)
        Q, H, r, c = fwd(matvec, krylov_depth, vector, matrix)
        fwd_ortho = Q.T @ Q - jnp.eye(len(Q.T))

        # Random values for the ``incoming VJPs''
        seed = 1
        key = jax.random.PRNGKey(seed)
        dQ, dH, dr, dc = exp_util.tree_random_like(key, (Q, H, r, c))

        # Backward pass
        kwargs_fwd = {"Q": Q, "H": H, "r": r, "c": c}
        kwargs_bwd = {"dQ": dQ, "dH": dH, "dr": dr, "dc": dc}
        _, mults = adj(matvec, matrix, **kwargs_fwd, **kwargs_bwd)
        received = mults["Lambda"].T @ Q
        expected = dH.T  # + jnp.triu(mults["Sigma"].T, 2)
        adj_ortho = received - expected

        # Plot
        epsilon = jnp.finfo(jnp.dtype(received)).eps
        vmin = int(jnp.log10(epsilon))

        # Context for the values:
        cond = jnp.linalg.cond(matrix)
        cond_vs_eps_val = jnp.log10(cond) + vmin
        cond_vs_eps = f"{round(cond_vs_eps_val, 2): 0.2f}"

        plot_kwargs = {
            "vmin": vmin,
            "vmax": -vmin,
            "cmap": cmap,
            "interpolation": "none",
        }
        colors = axes[reortho].imshow(magnitude(adj_ortho), **plot_kwargs)
        axes[reortho].set_aspect("equal")

    subfig.suptitle(f"{title} (cond. vs eps.: {cond_vs_eps})", fontsize="medium")
    cbar = subfig.colorbar(colors, ticks=(vmin, 0, -vmin))
    cbar.minorticks_on()
    cbar.ax.set_title("Log-error", fontsize="x-small")


directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}/figure.pdf")

plt.show()
