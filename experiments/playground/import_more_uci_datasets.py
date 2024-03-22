"""Import and plot UCI datasets."""

import matplotlib.pyplot as plt
from matfree_extensions import exp_util

X, y = exp_util.uci_household_electric(use_cache_if_possible=True)


print(X.shape)
print(y.shape)

print(X)


plt.plot(X[:, 0], y[:, 0], ".")
plt.show()
