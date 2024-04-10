"""Test equivalence of kernel parametrisations to GPyTorch."""

import gpytorch
import jax.numpy as jnp
import torch
from matfree_extensions import gp


def test_kernel_rbf():
    shape_in = (1,)
    shape_out = ()

    # Initialise the RBFKernel
    kernel_rbf = gpytorch.kernels.RBFKernel()
    k_torch = gpytorch.kernels.ScaleKernel(kernel_rbf)
    x_torch = torch.randn((3, *shape_in))
    y_torch = torch.randn((2, *shape_in))
    K_torch = _torch_to_array(k_torch(x_torch, y_torch).to_dense())

    # Initialise the jax version
    k_jax, p = gp.kernel_scaled_rbf(shape_in=shape_in, shape_out=shape_out)
    p["raw_lengthscale"] = _torch_to_array(kernel_rbf.raw_lengthscale)
    p["raw_outputscale"] = _torch_to_array(k_torch.raw_outputscale)
    x_jax = _torch_to_array(x_torch)
    y_jax = _torch_to_array(y_torch)
    K_jax = k_jax(**p)(x_jax, y_jax)

    # Compare that the kernel matrices are equal
    assert jnp.allclose(K_torch, K_jax)


def _torch_to_array(x):
    return jnp.asarray(x.detach().numpy())
