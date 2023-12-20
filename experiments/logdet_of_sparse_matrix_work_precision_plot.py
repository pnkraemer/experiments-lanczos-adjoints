import jax.numpy as jnp
import matplotlib.pyplot as plt
from tueplots import axes, bundles, markers

plt.rcParams.update(
    bundles.icml2022(column="full", family="sans-serif", nrows=1, ncols=2)
)
plt.rcParams.update(axes.lines())
plt.rcParams.update(markers.with_edge())
# plt.rcParams.update(axes.grid())

custom_vjp = jnp.load("./data/workprecision_dense_custom_vjp.npy", allow_pickle=True)[
    ()
]
custom_vjp_errors = jnp.asarray(custom_vjp["error"])
custom_vjp_wall_times = jnp.asarray(custom_vjp["wall_time"])

reference = jnp.load("./data/workprecision_dense_reference.npy", allow_pickle=True)[()]
reference_errors = jnp.asarray(reference["error"])
reference_wall_times = jnp.asarray(reference["wall_time"])


def reduce(x):
    return jnp.mean(x, axis=1)


mosaic = [
    ["value", "grad"],
]
fig, ax = plt.subplot_mosaic(mosaic, sharex=True, sharey=False, dpi=150)

ax["value"].set_title("Log-determinant: Value")
ax["value"].set_ylabel("Relative RMSE")
ax["grad"].set_title("Log-determinant: Gradient")
ax["value"].set_xlabel("Wall time [s]")
ax["grad"].set_xlabel("Wall time [s]")

custom_vjp_value, custom_vjp_grad = custom_vjp_errors
ax["value"].loglog(
    reduce(custom_vjp_wall_times),
    reduce(custom_vjp_value),
    marker="P",
    color="C0",
    linestyle="-",
    label="Custom VJP",
)
ax["grad"].loglog(
    reduce(custom_vjp_wall_times),
    reduce(custom_vjp_grad),
    marker="P",
    color="C0",
    linestyle="-",
    label="Custom VJP",
)

reference_value, reference_grad = reference_errors
ax["value"].loglog(
    reduce(reference_wall_times),
    reduce(reference_value),
    marker="o",
    color="C1",
    linestyle="-",
    label="AutoDiff",
)
ax["grad"].loglog(
    reduce(reference_wall_times),
    reduce(reference_grad),
    marker="o",
    color="C1",
    linestyle="-",
    label="AutoDiff",
)

ax["value"].legend()
ax["grad"].legend()
for a in ax.values():
    a.grid(which="minor", axis="both", linestyle="dotted")

plt.savefig("./figures/wp_sparse_matrix.pdf", dpi=150)
plt.show()
