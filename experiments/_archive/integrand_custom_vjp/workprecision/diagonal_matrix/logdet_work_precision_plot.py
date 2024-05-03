import jax.numpy as jnp
import matplotlib.pyplot as plt
from tueplots import axes, bundles, markers

plt.rcParams.update(
    bundles.icml2022(column="full", family="sans-serif", nrows=2, ncols=2)
)
plt.rcParams.update(axes.lines())
plt.rcParams.update(markers.with_edge())
# plt.rcParams.update(axes.grid())

custom_vjp = jnp.load("./data/workprecision_custom_vjp.npy", allow_pickle=True)[()]
custom_vjp_errors = jnp.asarray(custom_vjp["error"])
custom_vjp_wall_times = jnp.asarray(custom_vjp["wall_time"])

reference = jnp.load("./data/workprecision_reference.npy", allow_pickle=True)[()]
reference_errors = jnp.asarray(reference["error"])
reference_wall_times = jnp.asarray(reference["wall_time"])


mosaic = [
    ["custom_vjp-value", "custom_vjp-grad"],
    ["reference-value", "reference-grad"],
]
fig, ax = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, dpi=150)

ax["custom_vjp-value"].set_title("Logdet: Value")
ax["custom_vjp-value"].set_ylabel("Custom VJP | Rel. RMSE")
ax["custom_vjp-grad"].set_title("Logdet: Gradient")
ax["reference-value"].set_ylabel("Autodiff-of-SLQ | Rel. RMSE")
ax["reference-value"].set_xlabel("Wall time [s]")
ax["reference-grad"].set_xlabel("Wall time [s]")

custom_vjp_value, custom_vjp_grad = custom_vjp_errors
ax["custom_vjp-value"].loglog(
    custom_vjp_wall_times, custom_vjp_value, marker="X", color="C0", linestyle="None"
)
ax["custom_vjp-grad"].loglog(
    custom_vjp_wall_times, custom_vjp_grad, marker="P", color="C1", linestyle="None"
)

reference_value, reference_grad = reference_errors
ax["reference-value"].loglog(
    reference_wall_times, reference_value, marker="*", color="C2", linestyle="None"
)
ax["reference-grad"].loglog(
    reference_wall_times, reference_grad, marker="o", color="C3", linestyle="None"
)


for a in ax.values():
    a.grid(which="minor", axis="both", linestyle="dotted")

plt.savefig("./figures/wp_diagonal_matrix.pdf", dpi=150)
plt.show()
