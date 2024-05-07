"""Train an MLP on recovering the diffusion parameter."""

import argparse

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import optax
import tqdm
from matfree_extensions.util import pde_util

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
parser.add_argument("--method", type=str, required=True, help="Eg. 'arnoldi'")
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
print(args)


# Load data
path = f"./data/pde_wave/{args.resolution}x{args.resolution}"

inputs = jnp.load(f"{path}_data_inputs.npy")
targets = jnp.load(f"{path}_data_targets.npy")
parameter = jnp.load(f"{path}_data_parameter.npy")


key = jax.random.PRNGKey(args.seed)

# Create a mesh
xs_1d = jnp.linspace(0.0, 1.0, endpoint=True, num=args.resolution)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)

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


if args.method == "arnoldi":
    arnoldi_depth = 10
    expm = pde_util.expm_arnoldi(10)
    solve = pde_util.solver_expm(0.0, 1.0, vector_field, expm=expm)


mlp_features = [20, 20, 1]
mlp_init, mlp_apply = pde_util.model_mlp(
    mesh, mlp_features, activation=jnp.tanh, output_scale_raw=-5.0
)
key, subkey = jax.random.split(key, num=2)
variables_before, mlp_unflatten = mlp_init(subkey)
print(f"\nNumber of parameters: {variables_before.size}\n")


# Create a loss function
nugget = jnp.finfo(variables_before.dtype).eps
loss = pde_util.loss_mse_relative(nugget=nugget, reduce=jnp.sum)


@jax.jit
@jax.value_and_grad
def loss_value_and_grad(p, y0s, y1s):
    scale = mlp_apply(mlp_unflatten(p), mesh)
    approx = jax.vmap(lambda *a: solve(*a)[0], in_axes=(0, None))(y0s, scale)
    return loss(approx, targets=y1s)


variables_after = variables_before
optimizer = optax.adam(args.learning_rate)
opt_state = optimizer.init(variables_after)

progressbar = tqdm.tqdm(range(args.num_epochs))
progressbar.set_description(f"Loss {0:.3e}")
for _ in progressbar:
    try:
        loss, grad = loss_value_and_grad(variables_after, inputs, targets)
        updates, opt_state = optimizer.update(grad, opt_state)
        variables_after = optax.apply_updates(variables_after, updates)
        progressbar.set_description(f"Loss {loss:.3e}")
    except KeyboardInterrupt:
        break

fig, axes = plt.subplots(ncols=2)

img = axes[0].contourf(jnp.square(mlp_apply(mlp_unflatten(variables_after), mesh)))
plt.colorbar(img, ax=axes[0])
img = axes[1].contourf(jnp.square(parameter))
plt.colorbar(img, ax=axes[1])
plt.show()
