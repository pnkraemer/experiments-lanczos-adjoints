import functools
import os

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matfree import hutchinson, lanczos
from matfree_extensions.util import exp_util
from tueplots import axes

# todo: set fig- and fontsizes correctly
# todo: display the error of the mean
# todo: use a bigger matrix?

plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update({"font.size": 9})

mpl.rc("xtick", labelsize=8)
mpl.rc("ytick", labelsize=8)


ndata = 6
xs = jnp.linspace(0, 1, num=ndata)
K = jnp.exp(-0.5 * jnp.abs(xs[:, None] - xs[None, :]) ** 2)

eps = jnp.finfo(xs.dtype).eps


@hutchinson.integrand_trace
def matvec(v):
    return K @ v


@functools.partial(lanczos.integrand_spd, lambda x: jnp.log(jnp.maximum(x, eps)), 3)
def matvec_log_slq_3(v):
    return K @ v


@functools.partial(lanczos.integrand_spd, lambda x: 1 / jnp.maximum(x, eps), 3)
def matvec_inv_slq_3(v):
    return K @ v


@hutchinson.integrand_trace
def matvec_inv(v):
    return jnp.linalg.solve(K, v)


@hutchinson.integrand_trace
def matvec_log(v):
    return jax.scipy.linalg.funm(K, jnp.log) @ v


labels = {
    matvec: "$f(x)=x$",
    matvec_log: r"$f(x)=\log(x)$: Exact",
    matvec_log_slq_3: r"$f(x)=\log(x)$: Lanczos (3 matvecs)",
    matvec_inv: "$f(x)=1/x$: Exact",
    matvec_inv_slq_3: "$f(x)=1/x$: Lanczos (3 matvecs)",
}

styles_all = {
    "alpha": 0.7,
    "markeredgecolor": "black",
    "markeredgewidth": 0.5,
    "markersize": 4,
    "linewidth": 1.25,
}
styles = {
    matvec: {"color": "black", "linestyle": "dotted", "marker": "None", **styles_all},
    matvec_log: {"linestyle": "solid", "marker": "X", "color": "C0", **styles_all},
    matvec_log_slq_3: {
        "linestyle": "solid",
        "marker": "o",
        "color": "C2",
        **styles_all,
    },
    matvec_inv: {"linestyle": "dashed", "marker": "P", "color": "C1", **styles_all},
    matvec_inv_slq_3: {
        "linestyle": "dashed",
        "marker": "^",
        "color": "C3",
        **styles_all,
    },
}

nums = 3 ** jnp.arange(9.0)
plt.subplots(figsize=(3, 2.2), dpi=200)
# plt.title(f"Trace($f(A)$): {ndata}x{ndata} RBF kernel matrix", fontsize="medium")
key = jax.random.PRNGKey(1)

for mv in labels:
    means = []
    stds = []
    print(labels[mv])
    for num in nums:
        sampler = hutchinson.sampler_rademacher(xs, num=int(num))
        estimate = jax.jit(hutchinson.hutchinson(mv, sampler))

        key, subkey = jax.random.split(key, num=2)
        keys = jax.random.split(subkey, num=100)
        all_ = jax.lax.map(estimate, keys)
        means.append(jnp.mean(all_))
        stds.append(jnp.std(all_))
        print(num, jnp.std(all_), jnp.mean(all_))
    print()
    plt.loglog(nums, stds, label=labels[mv], **styles[mv])

plt.xlim((jnp.amin(nums), jnp.amax(nums)))
# plt.ylim((jnp.amin(nums), jnp.amax(nums)))
plt.grid(axis="both", which="major", linestyle="dotted")
plt.xlabel("No. samples")
plt.ylabel("Std. of trace-estimation")
plt.legend(fontsize="xx-small")
plt.tight_layout()


directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
plt.savefig(f"{directory_fig}trace_estimates.pdf")
plt.show()
