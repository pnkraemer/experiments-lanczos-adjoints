"""Tests for the extensions."""
from matfree import test_util
from matfree.backend import np

from matfree_extensions import slq_spd_with_grad


def test_slq_spd_with_grad():
    A = np.arange(0.0, 1.0, step=0.1) + 1.0
    print(A)
