"""Suite-sparse playground."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from matfree_extensions import exp_util

PATH = "./data/matrices/"

# nzbounds = (5_000, 15)
nzbounds = (None, 10_000)
sizebounds = (1_000, None)
exp_util.suite_sparse_download(
    path=PATH,
    limit=15,
    isspd=True,
    nzbounds=nzbounds,
    rowbounds=sizebounds,
    colbounds=sizebounds,
)

# matrices  = ["bcsstm09", "bcsstm21", "t2dal_e", "1138_bus"]
matrices = ["t2dal_e"]
fig, axes = plt.subplot_mosaic(
    [matrices], constrained_layout=True, figsize=(len(matrices) * 3, 3)
)

for matrix in matrices:
    M = exp_util.suite_sparse_load(matrix, path=PATH)
    print(jnp.linalg.slogdet(M.todense())[1])
    print(jnp.sum(jnp.log(jnp.abs(M.data))))
    axes[matrix].set_title(matrix)
    exp_util.plt_spy_coo(axes[matrix], M, cmap="viridis")

plt.show()
