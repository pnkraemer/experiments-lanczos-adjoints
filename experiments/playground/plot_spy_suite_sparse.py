"""Suite-sparse playground."""


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
M = exp_util.suite_sparse_load("bcsstm08", path=PATH)


fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 4))
ax.set_title(M)
exp_util.plt_spy_coo(ax, M, cmap="seismic")

plt.show()
