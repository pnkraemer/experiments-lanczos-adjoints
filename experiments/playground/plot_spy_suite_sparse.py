"""Suite-sparse playground."""


import matplotlib.pyplot as plt

from matfree_extensions import exp_util

PATH = "./data/matrices/"

exp_util.suite_sparse_download(path=PATH)
M = exp_util.suite_sparse_load("1138_bus", path=PATH)

fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 4))
exp_util.plt_spy_coo(ax, M, cmap="seismic")

plt.show()
