import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree_extensions.util import exp_util

directory = exp_util.matching_directory(__file__, "results/")

Ns = jnp.load(f"{directory}/Ns.npy")

ts_fwd = jnp.load(f"{directory}/ts_fwd.npy")
ts_bwd_custom = jnp.load(f"{directory}/ts_bwd_custom.npy")
ts_bwd_ad = jnp.load(f"{directory}/ts_bwd_ad.npy")


def plot_fun(a, b, **kw):
    return plt.semilogx(a[7:], b[7:], **kw)


plot_fun(Ns, ts_fwd, label="Forward pass")
plot_fun(Ns, ts_bwd_ad, label="Backward pass (AD)")
plot_fun(Ns, ts_bwd_custom, label="Backward pass (clever)")

plt.grid()
plt.legend()
plt.show()
