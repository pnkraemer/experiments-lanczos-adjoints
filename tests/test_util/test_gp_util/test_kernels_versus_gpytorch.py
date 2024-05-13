"""Test equivalence of the kernel parametrisations to GPyTorch."""

import functools

import gpytorch
import jax.numpy as jnp
import pytest_cases
import torch
from matfree_extensions.util import gp_util


def case_kernels_scaled_rbf():
    return gpytorch.kernels.RBFKernel, gp_util.kernel_scaled_rbf


def case_kernels_scaled_matern_12():
    matern = functools.partial(gpytorch.kernels.MaternKernel, nu=0.5)
    return matern, gp_util.kernel_scaled_matern_12


def case_kernels_scaled_matern_32():
    matern = functools.partial(gpytorch.kernels.MaternKernel, nu=1.5)
    return matern, gp_util.kernel_scaled_matern_32


@pytest_cases.parametrize_with_cases("kernels", cases=".")
def test_gp_kernels_match_gpy_torch_parametrisation(kernels):
    kernel_torch, kernel = kernels
    shape_in = (1,)
    shape_out = ()

    # Initialise the RBFKernel
    kernel_rbf = kernel_torch()
    k_torch = gpytorch.kernels.ScaleKernel(kernel_rbf)
    x_torch = torch.randn((3, *shape_in))
    y_torch = torch.randn((2, *shape_in))
    K_torch = _torch_to_array(k_torch(x_torch, y_torch).to_dense())

    # Initialise the jax version
    k_jax, p = kernel(shape_in=shape_in, shape_out=shape_out)
    p["raw_lengthscale"] = _torch_to_array(kernel_rbf.raw_lengthscale)
    p["raw_outputscale"] = _torch_to_array(k_torch.raw_outputscale)
    x_jax = _torch_to_array(x_torch)
    y_jax = _torch_to_array(y_torch)
    K_jax = gp_util.gram_matrix(k_jax(**p))(x_jax, y_jax)

    # Compare that the kernel matrices are equal
    assert jnp.allclose(K_torch, K_jax)


def _torch_to_array(x):
    return jnp.asarray(x.detach().numpy())
