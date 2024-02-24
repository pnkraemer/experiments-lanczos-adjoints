import matplotlib.pyplot as plt
from matfree_extensions import exp_util

# fetch dataset
inputs, targets = exp_util.uci_air_quality()

plt.plot(inputs[:100], targets[:100])
plt.show()
