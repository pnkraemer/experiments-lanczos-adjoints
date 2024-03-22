"""Test the kernel matrices."""

import jax.numpy as jnp
import pytest
import pytest_cases
from matfree_extensions import gp


def case_periodic():
    return gp.kernel_periodic


def case_matern_12():
    return gp.kernel_matern_12


def case_matern_32():
    return gp.kernel_matern_32


def case_kernel_quadratic_exponential():
    return gp.kernel_quadratic_exponential


def case_kernel_quadratic_rational():
    return gp.kernel_quadratic_rational


@pytest_cases.parametrize_with_cases("kernel", cases=".")
def test_gram_matrix_shape_is_as_expected(kernel):
    """Vector-valued kernels should yield correctly-shape kernel matrices."""
    k, p = kernel(shape_in=(1,), shape_out=(2,))
    x = jnp.arange(3)[:, None]
    y = jnp.arange(4)[:, None]

    K = k(**p)(x, y.T)
    assert K.shape == (2, 3, 4)


@pytest_cases.parametrize_with_cases("kernel", cases=".")
def test_raises_error_if_different_shapes(kernel):
    k, p = kernel(shape_in=(1,), shape_out=(2,))
    x = jnp.arange(3)[:, None]
    y = jnp.arange(4)[:, None, None]
    with pytest.raises(ValueError, match="different shapes"):
        _ = k(**p)(x, y.T)


@pytest_cases.parametrize_with_cases("kernel", cases=".")
def test_raises_value_error_if_shape_in_does_not_match_inputs(kernel):
    """Raise an error if shape_in does not match the input shapes."""
    k, p = kernel(shape_in=(2,), shape_out=(2,))
    x = jnp.arange(3)[:, None]
    y = jnp.arange(4)[:, None]

    with pytest.raises(ValueError, match="shape_in"):
        _ = k(**p)(x, y.T)
