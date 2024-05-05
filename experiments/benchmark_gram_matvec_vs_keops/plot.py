import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

plt.rcParams.update(figsizes.neurips2024(ncols=1, rel_width=0.4))
plt.rcParams.update(fontsizes.neurips2024(default_smaller=2))
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update(axes.grid())


directory = exp_util.matching_directory(__file__, "results/")

Ns = jnp.load(f"{directory}/Ns.npy")

ts_fwd = jnp.load(f"{directory}/ts_fwd.npy")
ts_bwd_custom = jnp.load(f"{directory}/ts_bwd_custom.npy")
ts_bwd_ad = jnp.load(f"{directory}/ts_bwd_ad.npy")


def plot_fun(a, b, **kw):
    return plt.loglog(a[: len(b)], b, **kw, base=2)


plt.subplots(dpi=200)

plot_fun(Ns, ts_fwd, label="Forward pass", linestyle="dashed", color="black")
plot_fun(Ns, ts_bwd_ad, label="Backprop")
plot_fun(Ns, ts_bwd_custom, label="Custom VJP")

plt.xlabel("Data set size", fontsize="small")
plt.ylabel("Run time (sec)", fontsize="small")
plt.title("Gradients of kernel-vector products", fontsize="small")
plt.grid()
plt.ylim((0.08, 20.0))
plt.legend(fontsize="x-small")


directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}figure.pdf")
plt.show()
