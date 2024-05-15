"""Create a data set."""

import argparse
import functools
import os

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import tqdm
from matfree_extensions.util import gp_util, pde_util
from tueplots import axes

plt.rcParams.update(axes.lines())
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_data", type=int, required=True, help="Eg. 10, 100, ...")
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
parser.add_argument("--lanczos_rank", type=int, default=32)
args = parser.parse_args()
print(args)


# Create the mesh
xs_1d = jnp.linspace(0.0, 1.0, endpoint=True, num=args.resolution)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)

# Prepare randomness
key = jax.random.PRNGKey(seed=args.seed)
gram = gp_util.gram_matvec()
xs = mesh.reshape((2, -1)).T
mean = jnp.zeros_like(mesh[0]).reshape((-1,))

# Create a sampler for a Gaussian random field
kernel_p, _ = gp_util.kernel_scaled_rbf(shape_in=(2,), shape_out=())
kernel = kernel_p(raw_lengthscale=-0.75, raw_outputscale=-10.0)
matvec = functools.partial(gram(kernel), xs, xs)
sample = pde_util.sampler_lanczos(
    mean=mean, cov_matvec=matvec, num=1, lanczos_rank=args.lanczos_rank
)

# Sample the initial parameter
key, subkey = jax.random.split(key, num=2)
samples = sample(subkey)
parameter = samples.reshape(mesh[0].shape)

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


# Set up a GP for the initial condition
kernel_p, _ = gp_util.kernel_scaled_rbf(shape_in=(2,), shape_out=())
kernel = kernel_p(raw_lengthscale=0.0, raw_outputscale=0.0)
matvec = functools.partial(gram(kernel), xs, xs)
sample = pde_util.sampler_lanczos(
    mean=mean, cov_matvec=matvec, num=1, lanczos_rank=args.lanczos_rank
)

# Choose a solver
solver_kwargs = {"num_steps": 128, "method": "dopri8", "adjoint": "direct"}
solver = pde_util.solver_diffrax(0.0, 1.0, vector_field, **solver_kwargs)
solver = jax.jit(solver)

inputs = []
targets = []

for _ in tqdm.tqdm(range(args.num_data)):
    key, subkey = jax.random.split(key, num=2)
    samples = sample(subkey)
    y0 = samples.reshape(mesh[0].shape)

    key, subkey = jax.random.split(key, num=2)
    samples = sample(subkey)
    dy0 = samples.reshape(mesh[0].shape)

    init = jnp.stack([y0, dy0])
    final, _aux = solver(init, parameter)

    inputs.append(init)
    targets.append(final)


# Make directories
directory = "./data/pde_wave/"
os.makedirs(directory, exist_ok=True)
path = f"{directory}{args.resolution}x{args.resolution}"

# Save the inputs and targets
inputs = jnp.stack(inputs)
targets = jnp.stack(targets)
jnp.save(f"{path}_data_inputs.npy", inputs)
jnp.save(f"{path}_data_targets.npy", targets)
jnp.save(f"{path}_data_parameter.npy", parameter)


print("\nPlotting the inputs and targets...", end="")
num = jnp.minimum(3, args.num_data)  # plot only a few data points
num = int(num)

figsize = (2 * num, 2)
fig, axes = plt.subplots(nrows=2, ncols=num, figsize=figsize, constrained_layout=True)

for ax, ins, outs in zip(axes.T[:num], inputs[:num], targets[:num]):
    img = ax[0].contourf(*mesh, ins[0], cmap="bone")
    cb = plt.colorbar(img, ax=ax[0])
    cb.ax.tick_params(labelsize="xx-small")
    ax[0].tick_params(labelsize="x-small")

    img = ax[1].contourf(*mesh, outs[1], cmap="pink")
    cb = plt.colorbar(img, ax=ax[1])
    cb.ax.tick_params(labelsize="xx-small")
    ax[1].tick_params(labelsize="x-small")

axes[0][0].set_ylabel("Inputs")
axes[1][0].set_ylabel("Targets")

plt.savefig(f"{path}_plot_data.pdf")
print("done.")

print("Plotting the parameter...", end="")
plt.subplots()
img = plt.contourf(parameter)
plt.colorbar(img)
plt.savefig(f"{path}_plot_parameter.pdf")
print("done.\n")
