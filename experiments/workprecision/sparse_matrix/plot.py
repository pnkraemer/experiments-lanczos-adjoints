import jax.numpy as jnp
import matplotlib.pyplot as plt
from tueplots import axes, bundles, markers

from matfree_extensions import exp_util

plt.rcParams.update(
    bundles.icml2022(column="full", family="sans-serif", nrows=2, ncols=2)
)
plt.rcParams.update(axes.lines())
plt.rcParams.update(markers.with_edge())


directory = exp_util.matching_directory(__file__, "data/")

eigvals = jnp.load(f"{directory}/eigvals.npy")

custom_vjp = jnp.load(f"{directory}/custom_vjp.npy", allow_pickle=True)[()]
custom_vjp_errors = jnp.asarray(custom_vjp["error"])
custom_vjp_wall_times = jnp.asarray(custom_vjp["wall_time"])

reference = jnp.load(f"{directory}/reference.npy", allow_pickle=True)[()]
reference_errors = jnp.asarray(reference["error"])
reference_wall_times = jnp.asarray(reference["wall_time"])


def plot(
    axis, times, values, marker, color, linestyle, label, num_stdevs=1, alpha=0.25
):
    ts = reduce_t(times)
    ms, stds = reduce_s(values)
    axis.loglog(ts, ms, marker=marker, color=color, linestyle=linestyle, label=label)
    axis.fill_between(
        ts, ms - num_stdevs * stds, ms + num_stdevs * stds, color=color, alpha=alpha
    )


def reduce_t(x):
    return jnp.mean(x, axis=1)


def reduce_s(x):
    return jnp.mean(x, axis=1), jnp.std(x, axis=1)


mosaic = [["value", "grad"], ["eigvals", "eigvals"]]
fig, ax = plt.subplot_mosaic(mosaic, sharex=False, sharey=False, dpi=150)

ax["value"].set_title("Log-determinant: Value")
ax["value"].set_ylabel("Relative RMSE")
ax["grad"].set_title("Log-determinant: Gradient")
ax["value"].set_xlabel("Wall time [s]")
ax["grad"].set_xlabel("Wall time [s]")

custom_vjp_value, custom_vjp_grad = custom_vjp_errors
plot(
    ax["value"],
    custom_vjp_wall_times,
    custom_vjp_value,
    marker="P",
    color="C0",
    linestyle="-",
    label="Custom VJP",
)
plot(
    ax["grad"],
    custom_vjp_wall_times,
    custom_vjp_grad,
    marker="P",
    color="C0",
    linestyle="-",
    label="Custom VJP",
)


reference_value, reference_grad = reference_errors
plot(
    ax["value"],
    reference_wall_times,
    reference_value,
    marker="o",
    color="C1",
    linestyle="-",
    label="AutoDiff",
)
plot(
    ax["grad"],
    reference_wall_times,
    reference_grad,
    marker="o",
    color="C1",
    linestyle="-",
    label="AutoDiff",
)


ax["value"].legend()
ax["grad"].legend()
for a in ax.values():
    a.grid(which="minor", axis="both", linestyle="dotted")


ax["eigvals"].set_title("Eigenvalues of the matrix")
ax["eigvals"].grid(axis="y", which="major")
ax["eigvals"].semilogy(eigvals, linestyle="None", marker="X", markersize=4, alpha=0.6)

plt.savefig("./figures/wp_sparse_matrix.pdf", dpi=150)
plt.show()
