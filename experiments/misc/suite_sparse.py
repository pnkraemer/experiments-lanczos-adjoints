"""Suite-sparse playground."""


import matplotlib.pyplot as plt

from matfree_extensions import test_util

PATH = "./data/matrices/"

test_util.suite_sparse_download(path=PATH)
M = test_util.suite_sparse_load("494_bus", path=PATH)

fig, ax = plt.subplots(ncols=7)
ax[0].spy(M.todense())
for i, a in enumerate(ax[1:]):
    a.spy(M.todense(), markersize=i + 1)
plt.show()
