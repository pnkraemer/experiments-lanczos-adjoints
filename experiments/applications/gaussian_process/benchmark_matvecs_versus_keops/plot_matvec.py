import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util

labels = [
    "dim",
    "Matfree (via JAX's map)",
    "Matfree (via JAX's vmap)",
    "GPyTorch (via pykeops)",
]
labels = labels  # + [f"Matfree (via map-over-vmap; {bnum} batches)" for bnum in [1, 16, 256]]

print(labels)

directory = exp_util.matching_directory(__file__, "results/")


LABEL = f"num_runs_{3}_data_size_{40_000}"

results = {}
for x in labels:
    path = f"{directory}matvec_per_data_dim_{LABEL}_{x}.npy"
    results[x] = jnp.load(path)

print(results)
inputs = results["dim"]
for key, value in results.items():
    if key != "dim":
        print(inputs.shape)
        print(value.shape)
        if " map" in key:
            plt.semilogy(inputs[: len(value)], jnp.amin(value, axis=1), label=key)
        else:
            plt.semilogy(inputs, jnp.amin(value, axis=1), label=key)
        # plt.loglog(inputs, jnp.amax(value, axis=1), label=key)

plt.grid(which="major", linestyle="dotted")
plt.legend()
plt.show()
print(results)

################################################################################


labels = [
    "num",
    "Matfree (via JAX's map)",
    "Matfree (via JAX's vmap)",
    "GPyTorch (via pykeops)",
]
labels = labels  # + [f"Matfree (via map-over-vmap; {bnum} batches)" for bnum in [1, 16, 256]]

print(labels)

LABEL = f"num_runs_{3}_data_dim_{1}"

results = {}
for x in labels:
    path = f"{directory}matvec_per_data_size_{LABEL}_{x}.npy"
    results[x] = jnp.load(path)

print(results)
inputs = results["num"]
for key, value in results.items():
    if key != "num":
        print(inputs.shape)
        print(value.shape)
        if " map" in key:
            plt.loglog(inputs[: len(value)], jnp.amin(value, axis=1), label=key)
        else:
            plt.loglog(inputs, jnp.amin(value, axis=1), label=key)
        # plt.loglog(inputs, jnp.amax(value, axis=1), label=key)

plt.grid(which="major", linestyle="dotted")
plt.legend()
plt.show()
print(results)
