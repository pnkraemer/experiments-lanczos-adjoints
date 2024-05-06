"""Create a data set."""

import argparse
import os
import pickle
import time
import warnings

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import optax
import tqdm
from matfree_extensions.util import exp_util, gp_util, pde_util


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_data", type=int, required=True, help="Eg. 10, 100, ...")
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
args = parser.parse_args()
print(args)

# Prepare randomness
key = jax.random.PRNGKey(seed=args.seed)

# Create the mesh
xs_1d = jnp.linspace(0.0, 1.0, endpoint=True, num=args.resolution)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)


# Create a Gaussian random field
kernel_p, _ = gp_util.kernel_scaled_rbf(shape_in=(2,), shape_out=())
kernel = kernel_p(raw_lengthscale=-0.75, raw_outputscale=-5.0)
xs = mesh.reshape((2, -1)).T
K = gp_util.gram_matrix(kernel)(xs, xs)


# Prepare sampling 
# Use eigh() so we can handle almost singular matrices
(w, v) = jnp.linalg.eigh(K)
w = jnp.maximum(0.0, w)  # clamp to allow sqrts
factor = v * jnp.sqrt(w[..., None, :])

# Sample a true parameter
key, subkey = jax.random.split(key, num=2)
eps = jax.random.normal(subkey, shape=xs[:, 0].shape)
parameter = (factor @ eps).reshape(mesh[0].shape)
parameter = jnp.square(parameter)  # ensure nonnegativity

# Discretise the PDE
dx_space = jnp.diff(xs_1d)[0]
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_wave_anisotropic(
    parameter, constrain=jnp.square, stencil=stencil, boundary=boundary
)


def vector_field(x, p):
    """Evaluate the PDE dynamics."""
    return pde_rhs(scale=p)(x)


# Choose a solver
solver_kwargs = {"num_steps":128, "method":"dopri8", "adjoint":"direct"}
solver = pde_util.solver_diffrax(0., 1., vector_field, **solver_kwargs)
solver = jax.jit(solver)

inputs = []
targets = []

for _ in tqdm.tqdm(range(args.num_data)):
    key, subkey = jax.random.split(key, num=2)
    eps = jax.random.normal(subkey, shape=xs[:, 0].shape)
    y0 = (factor @ eps).reshape(mesh[0].shape)

    key, subkey = jax.random.split(key, num=2)
    eps = jax.random.normal(subkey, shape=xs[:, 0].shape)
    dy0 = (factor @ eps).reshape(mesh[0].shape)

    init = jnp.stack([y0, dy0])
    final = solver(init, parameter)

    inputs.append(init)
    targets.append(final)





# Make directories
directory = exp_util.matching_directory(__file__, "data/")
os.makedirs(directory, exist_ok=True)
path = f"{directory}{args.resolution}x{args.resolution}"

# Save the inputs and targets
inputs = jnp.stack(inputs)
targets = jnp.stack(targets)
jnp.save(f"{path}_data_inputs.npy", inputs)
jnp.save(f"{path}_data_targets.npy", targets)
jnp.save(f"{path}_data_parameter.npy", parameter)


print("\nPlotting the inputs and targets...", end="")
num = jnp.minimum(8, args.num_data)  # plot only a few data points
num = int(num)

figsize = (2*num, 2)
fig, axes = plt.subplots(nrows=2, ncols=num, figsize=figsize, constrained_layout=True)

for ax, ins, outs in zip(axes.T[:num], inputs[:num], targets[:num]):
    img = ax[0].contourf(ins[0])
    plt.colorbar(img, ax=ax[0])

    img = ax[1].contourf(outs[1])
    plt.colorbar(img, ax=ax[1])


axes[0][0].set_xlabel("Inputs")
axes[1][0].set_ylabel("Targets")

plt.savefig(f"{path}_plot_data.pdf")
print("done.")

print("Plotting the parameter...", end="")
plt.subplots()
img = plt.contourf(parameter)
plt.colorbar(img)
plt.savefig(f"{path}_plot_parameter.pdf")
print("done.\n")