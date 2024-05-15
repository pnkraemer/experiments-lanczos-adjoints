"""Measure the loss of accuracy with and without re-projection."""

import os

import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from matfree_extensions import arnoldi
from matfree_extensions.util import exp_util
from tueplots import axes, figsizes, fontsizes

plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update(figsizes.iclr2024(rel_width=0.4, height_to_width_ratio=1.25))
plt.rcParams.update(fontsizes.iclr2024(default_smaller=2))

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(1)
key_A, key_v = jax.random.split(key, num=2)

plt.subplots(dpi=200)
setups = {
    (True, "full", "full"): "Adj. w/ re-proj.",
    (True, "full", "none"): "Adj. w/o re-proj.",
    (False, "full", None): "Backprop",
}
styles_all = {"linewidth": 1.0, "alpha": 0.8}
styles = {
    (True, "full", "full"): {
        "linestyle": "dashed",
        "zorder": 100,
        "color": "C0",
        **styles_all,
    },
    (True, "full", "none"): {"linestyle": "solid", "color": "C1", **styles_all},
    (False, "full", None): {"linewidth": 4, "color": "gray", "alpha": 0.35},
}
for (custom, reortho, match), label in tqdm.tqdm(setups.items()):
    ns = jnp.arange(1, 9, step=1)
    loss = []
    for n in ns:
        n = int(n)
        A = exp_util.hilbert(n)

        algorithm = arnoldi.hessenberg(
            lambda s, p: p @ s, n, reortho=reortho, reortho_vjp=match, custom_vjp=False
        )
        flat, unflatten = jax.flatten_util.ravel_pytree(A)

        v = jax.random.normal(key_v, shape=(n,))
        Q, *_ = algorithm(v, A)

        @jax.jit
        @jax.jacrev
        def identity(x):
            a = unflatten(x)
            q, h, r, c = algorithm(v, a)
            return jax.flatten_util.ravel_pytree(q @ h @ q.T)[0]

        diff = jnp.eye(len(flat)) - identity(flat)
        error = jnp.sqrt(jnp.mean(diff**2))
        loss.append(error)

    print(loss)
    loss = jnp.asarray(loss)
    plt.semilogy(ns, loss, label=label, **styles[(custom, reortho, match)])

plt.legend(fontsize="xx-small")
plt.xlabel("Hilbert matrix size", fontsize="small")
plt.ylabel("Loss of accuracy", fontsize="small")
plt.ylim((1e-18, 1e0))

directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
plt.savefig(f"{directory_fig}accuracy_loss.pdf")

plt.show()
