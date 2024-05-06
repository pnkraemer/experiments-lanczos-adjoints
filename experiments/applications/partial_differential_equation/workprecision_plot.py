"""Plot work vs. precision of value_and_grad of matrix exponentials."""

# todo: run in large scale and plot results
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util

directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)

fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

methods = [
    "arnoldi",
    "diffrax:euler+backsolve",
    "diffrax:heun+recursive_checkpoint",
    "diffrax:dopri5+backsolve",
    "diffrax:tsit5+recursive_checkpoint",
]
for method in methods:
    num_matvecs = jnp.load(f"{directory}/wp_{method}_Ns.npy")
    ts_all = jnp.load(f"{directory}/wp_{method}_ts.npy")
    errors_fwd = jnp.load(f"{directory}/wp_{method}_errors_fwd.npy")
    errors_rev = jnp.load(f"{directory}/wp_{method}_errors_rev.npy")

    eps = 0.0 * jnp.finfo(errors_fwd).eps ** 2
    print()
    print(method)
    print("MVs", num_matvecs)
    print("fwd", errors_fwd)
    print("rev", errors_rev)
    print()

    axes[0][0].semilogy(ts_all, errors_fwd + eps, label=method)
    axes[0][1].semilogy(ts_all, errors_rev + eps, label=method)
    axes[1][0].semilogy(num_matvecs, errors_fwd + eps, label=method)
    axes[1][1].semilogy(num_matvecs, errors_rev + eps, label=method)

axes[0][0].set_xlabel("Time (sec)")
axes[0][0].set_ylabel("Rel. MSE (value)")
axes[0][0].legend(fontsize="x-small")
axes[0][0].grid()

axes[0][1].set_xlabel("Time (sec)")
axes[0][1].set_ylabel("Rel. MSE (gradient)")
axes[0][1].legend(fontsize="x-small")
axes[0][1].grid()

axes[1][0].set_xlabel("# Matvecs")
axes[1][0].set_ylabel("Rel. MSE (value)")
axes[1][0].legend(fontsize="x-small")
axes[1][0].grid()

axes[1][1].set_xlabel("# Matvecs")
axes[1][1].set_ylabel("Rel. MSE (gradient)")
axes[1][1].legend(fontsize="x-small")
axes[1][1].grid()


directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}workprecision.pdf")
plt.show()
