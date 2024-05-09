"""Tests for selected logpdf functionality."""

from typing import Callable

import jax
import jax.numpy as jnp
import pytest_cases
from matfree import test_util
from matfree_extensions.util import bnn_util, gp_util


@pytest_cases.case
def case_logpdf_cholesky():
    return bnn_util.logpdf_cholesky()


@pytest_cases.parametrize_with_cases("implementation", ".")
def test_value_versus_cholesky(implementation: Callable, n=10):
    key = jax.random.PRNGKey(1)
    key1, key2, key3 = jax.random.split(key, num=3)

    mean = jax.random.normal(key1, shape=(n,))
    cov_eig = 0.1 + jax.random.uniform(key2, shape=(n,))
    cov = test_util.symmetric_matrix_from_eigenvalues(cov_eig)
    y = jax.random.normal(key3, shape=(n,))

    reference = gp_util.logpdf_cholesky()
    truth, _info = reference(y, mean=mean, cov=lambda s: cov @ s)
    approx, _info = implementation(y, mean=mean, cov=lambda s: cov @ s)
    assert jnp.allclose(truth, approx)
