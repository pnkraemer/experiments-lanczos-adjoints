"""Plot work vs. precision of value_and_grad of matrix exponentials."""

# todo: run in large scale and plot results
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util

directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)

fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)


labels = {
    "arnoldi": "Arnoldi/Adjoints",
    "diffrax:euler+backsolve": "Euler/Backsolve (Diffrax)",
    "diffrax:heun+recursive_checkpoint": "Heun/Autodiff (Diffrax)",
    "diffrax:dopri5+backsolve": "Dopri5/Backsolve (Diffrax)",
    "diffrax:tsit5+recursive_checkpoint": "Tsit5/Autodiff (Diffrax)",
}
methods = list(labels.keys())

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

    idx = num_matvecs >= 5

    axes[0][0].semilogy(jnp.amin(ts_all[idx:], axis=1), errors_fwd[idx:] + eps, label=labels[method])
    axes[0][1].semilogy(jnp.amin(ts_all[idx:], axis=1), errors_rev[idx:] + eps, label=labels[method])
    axes[1][0].semilogy(num_matvecs[idx:], errors_fwd[idx:] + eps, label=labels[method])
    axes[1][1].semilogy(num_matvecs[idx:], errors_rev[idx:] + eps, label=labels[method])

axes[0][0].set_xlabel("Time (sec)")
axes[0][0].set_ylabel("Rel. MSE (value)")
axes[0][0].legend(fontsize="xx-small")
axes[0][0].grid()

axes[0][1].set_xlabel("Time (sec)")
axes[0][1].set_ylabel("Rel. MSE (gradient)")
axes[0][1].legend(fontsize="xx-small")
axes[0][1].grid()

axes[1][0].set_xlabel("# Matvecs")
axes[1][0].set_ylabel("Rel. MSE (value)")
axes[1][0].legend(fontsize="xx-small")
axes[1][0].grid()

axes[1][1].set_xlabel("# Matvecs")
axes[1][1].set_ylabel("Rel. MSE (gradient)")
axes[1][1].legend(fontsize="xx-small")
axes[1][1].grid()



directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}workprecision.pdf")
plt.show()
