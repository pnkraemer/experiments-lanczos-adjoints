"""Train an MLP on recovering the diffusion parameter."""

import argparse
import os
import pickle
import time

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import optax
import tqdm
from matfree_extensions.util import exp_util, pde_util

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
parser.add_argument("--method", type=str, required=True, help="Eg. 'arnoldi'")
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
print(args)


print("\nLoading data...", end=" ")
path = f"./data/pde_wave/{args.resolution}x{args.resolution}"
inputs_all = jnp.load(f"{path}_data_inputs.npy")
targets_all = jnp.load(f"{path}_data_targets.npy")
parameter = jnp.load(f"{path}_data_parameter.npy")
ndata = len(inputs_all)
ntest = int(0.2 * ndata)
print("done.")

ntrain = ndata - ntest
print(f"Splitting data into {ntrain}/{ntest} train/test points...", end=" ")
test_inputs, test_targets = inputs_all[:ntest], targets_all[:ntest]
inputs, targets = inputs_all[ntest:], targets_all[ntest:]
print("done.")


print("Creating a mesh...", end=" ")
xs_1d = jnp.linspace(0.0, 1.0, endpoint=True, num=args.resolution)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)
print("done.")

print("Discretising the PDE...", end=" ")
dx_space = jnp.diff(xs_1d)[0]
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_wave_anisotropic(
    parameter, constrain=jnp.square, stencil=stencil, boundary=boundary
)


def vector_field(x, p):
    """Evaluate the PDE dynamics."""
    return pde_rhs(scale=p)(x)


print("done.")


print(f"Setting up solver {args.method}...", end=" ")
if args.method == "arnoldi":
    arnoldi_depth = 10
    expm = pde_util.expm_arnoldi(10)
    solve = pde_util.solver_expm(0.0, 1.0, vector_field, expm=expm)

elif args.method == "diffrax:euler+backsolve":
    method, adjoint = "euler", "backsolve"
    kwargs = {"num_steps": 10, "method": method, "adjoint": adjoint}
    solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

elif args.method == "diffrax:heun+recursive_checkpoint":
    method, adjoint = "heun", "recursive_checkpoint"
    kwargs = {"num_steps": 10, "method": method, "adjoint": adjoint}
    solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

elif args.method == "diffrax:tsit5+recursive_checkpoint":
    method, adjoint = "tsit5", "recursive_checkpoint"
    kwargs = {"num_steps": 10, "method": method, "adjoint": adjoint}
    solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

elif args.method == "diffrax:dopri5+backsolve":
    method, adjoint = "dopri5", "backsolve"
    kwargs = {"num_steps": 10, "method": method, "adjoint": adjoint}
    solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

else:
    msg = f"The method {args.method} is not supported."
    raise ValueError(msg)
print("done.")

print("Setting up the MLP...", end=" ")
key = jax.random.PRNGKey(args.seed)
kwargs = {"output_scale_raw": -5.0, "activation": jnp.tanh}
mlp_init, mlp_apply = pde_util.model_mlp(mesh, [20, 20, 1], **kwargs)
variables_before, mlp_unflatten = mlp_init(key)
print(f"with {variables_before.size} parameters.")


print("Creating a loss function...", end=" ")
nugget = jnp.finfo(variables_before.dtype).eps
error = pde_util.loss_mse_relative(nugget=nugget, reduce=jnp.sum)


@jax.jit
@jax.value_and_grad
def loss_value_and_grad(p, y0s, y1s):
    """Evaluate the loss over all input/output pairs."""
    scale = mlp_apply(mlp_unflatten(p), mesh)
    approx = jax.vmap(lambda *a: solve(*a)[0], in_axes=(0, None))(y0s, scale)
    return error(approx, targets=y1s)


print("done.")


print("Setting up an optimiser...", end=" ")
variables_after = variables_before
optimizer = optax.adam(args.learning_rate)
opt_state = optimizer.init(variables_after)
print("done.")

print("Precompiling the value-and-grad...", end=" ")
loss, grad = loss_value_and_grad(variables_after, inputs, targets)
loss.block_until_ready()
grad.block_until_ready()
print("done.")

print("Training...\n")
progressbar = tqdm.tqdm(range(args.num_epochs))
progressbar.set_description(f"Loss {0:.3e}")
t0 = time.perf_counter()
for _ in progressbar:
    loss, grad = loss_value_and_grad(variables_after, inputs, targets)
    updates, opt_state = optimizer.update(grad, opt_state)
    variables_after = optax.apply_updates(variables_after, updates)
    progressbar.set_description(f"Loss {loss:.3e}")

t1 = time.perf_counter()
scale_after = mlp_apply(mlp_unflatten(variables_after), mesh)

print("\nEvaluating the error metrics...", end=" ")
runtime = (t1 - t0) / args.num_epochs
rmse = error(jnp.abs(scale_after), targets=jnp.abs(parameter))
loss, _grad = loss_value_and_grad(variables_after, test_inputs, test_targets)
print("\n\tRuntime per epoch:", runtime)
print("\tRMSE (param):", rmse)
print("\tLoss:", loss)
print("done.\n")


print("Saving results...", end=" ")
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)
path = f"{directory}{args.resolution}x{args.resolution}_{args.method}"
stats = {"runtime": runtime, "rmse_param": rmse, "loss": loss}
with open(f"{path}_stats.pkl", "wb") as handle:
    pickle.dump(stats, handle)

jnp.save(f"{path}_parameter.npy", scale_after)
print("done.")

print("Plotting results... todo: remove soon", end=" ")
fig, axes = plt.subplots(ncols=2)
img = axes[0].contourf(jnp.abs(scale_after))
plt.colorbar(img, ax=axes[0])
img = axes[1].contourf(jnp.abs(parameter))
plt.colorbar(img, ax=axes[1])
plt.show()
print("done.")
