"""Import and plot UCI datasets."""

from matfree_extensions.util import uci_util

print("\nRoad network:")
inputs, targets = uci_util.uci_road_network()
print(inputs.shape, targets.shape)


print("\nSong:")
inputs, targets = uci_util.uci_song()
print(inputs.shape, targets.shape)

print("\nAir quality:")
inputs, targets = uci_util.uci_air_quality()
print(inputs.shape, targets.shape)


print("\nBike sharing:")
inputs, targets = uci_util.uci_bike_sharing()
print(inputs.shape, targets.shape)


print("\nKEGG undirected:")
inputs, targets = uci_util.uci_kegg_undirected()
print(inputs.shape, targets.shape)


print("\nParkinson:")
inputs, targets = uci_util.uci_parkinson()
print(inputs.shape, targets.shape)


print("\nProtein:")
inputs, targets = uci_util.uci_protein()
print(inputs.shape, targets.shape)


print("\nSGEMM:")
inputs, targets = uci_util.uci_sgemm()
print(inputs.shape, targets.shape)


print("\nConcrete:")
inputs, targets = uci_util.uci_concrete()
print(inputs.shape, targets.shape)


print("\nPower plant:")
inputs, targets = uci_util.uci_power_plant()
print(inputs.shape, targets.shape)
