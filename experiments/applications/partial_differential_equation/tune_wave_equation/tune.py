import argparse
import os

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import numpy as onp  # for matplotlib manipulations  # noqa: ICN001
import optax
import tqdm
from matfree_extensions.util import exp_util, gp_util, pde_util

# todo: add a "naive" matrix exponential solver

# todo: compute the error of the simulation
#  to ensure they are all equally accurate

# todo: quantify the reconstruction errors a little bit

# todo: run for different solvers

# todo: figure out what to plot...

# todo: verify somehow that we do solve a wave equation!

# Make directories
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)

# Parse arguments
parser = argparse.ArgumentParser()
methods = ["euler", "arnoldi", "diffrax_tsit5", "diffrax_euler", "diffrax_heun"]
parser.add_argument("--method", help=f"One of {methods}", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_matvecs", type=int, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--num_dx_points", type=int, required=True)
args = parser.parse_args()


# Set parameters
pde_t0, pde_t1 = 0.0, 1.0
mlp_features = [20, 20, 1]
mlp_activation = jax.nn.tanh
optimizer = optax.adam(1e-3)

# Process the parameters
key = jax.random.PRNGKey(args.seed)
solve_ts = jnp.linspace(pde_t0, pde_t1, num=args.num_matvecs, endpoint=True)
arnoldi_depth = args.num_matvecs


# Discretise the space-domain
xs_1d = jnp.linspace(0.0, 1.0, endpoint=True, num=args.num_dx_points)
dx_space = jnp.diff(xs_1d)[0]
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)
print(f"Number of points: {mesh.size // 2}")


def constrain(arg):
    """Constrain the PDE-scale to strictly positive values."""
    return 0.1 * arg**2


# Sample a Gaussian random field as a "true" scale
grf_xs = mesh.reshape((2, -1)).T
grf_kernel, _ = gp_util.kernel_scaled_rbf(shape_in=(2,), shape_out=())
kernel_fun = grf_kernel(raw_lengthscale=-0.75, raw_outputscale=-4.0)
grf_K = gp_util.gram_matrix(kernel_fun)(grf_xs, grf_xs)
grf_K += 1e-6 * jnp.eye(len(grf_K))
grf_cholesky = jnp.linalg.cholesky(grf_K)

key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_scale = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)


# Initial condition
key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_init = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)

key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_init_diff = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)
y0 = jnp.stack([grf_init, grf_init_diff])


# Discretise the PDE dynamics with method of lines (MOL)
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_wave_anisotropic(
    grf_scale, constrain=constrain, stencil=stencil, boundary=boundary
)


# Create the data/targets


def vector_field(x):
    """Evaluate the PDE dynamics."""
    return pde_rhs(scale=grf_scale)(x)


solve_ts_data = jnp.linspace(pde_t0, pde_t1, endpoint=True, num=10_000)
target_solve = pde_util.solver_euler_fixed_step(solve_ts_data, vector_field)
target_y1 = target_solve(y0)

# Build an approximate model
mlp_init, mlp_apply = pde_util.model_mlp(mesh, mlp_features, activation=mlp_activation)
key, subkey = jax.random.split(key, num=2)
variables_before, mlp_unflatten = mlp_init(subkey)
print(f"Number of parameters: {variables_before.size}")


def vector_field_mlp(x, p):
    """Evaluate the MLP-parametrised PDE dynamics."""
    scale = mlp_apply(mlp_unflatten(p), mesh)
    return pde_rhs(scale=scale)(x)


# Create a loss function
if args.method == "arnoldi":
    expm = pde_util.expm_arnoldi(arnoldi_depth)
    approx_solve = pde_util.solver_arnoldi(pde_t0, pde_t1, vector_field_mlp, expm=expm)
elif args.method == "euler":
    approx_solve = pde_util.solver_euler_fixed_step(solve_ts, vector_field_mlp)
elif args.method == "diffrax_tsit5":
    approx_solve = pde_util.solver_diffrax(
        pde_t0,
        pde_t1,
        vector_field_mlp,
        num_steps=args.num_matvecs // 5,
        method="tsit5",
    )
elif args.method == "diffrax_euler":
    approx_solve = pde_util.solver_diffrax(
        pde_t0, pde_t1, vector_field_mlp, num_steps=args.num_matvecs, method="euler"
    )
elif args.method == "diffrax_heun":
    approx_solve = pde_util.solver_diffrax(
        pde_t0, pde_t1, vector_field_mlp, num_steps=args.num_matvecs // 2, method="heun"
    )
else:
    msg = f"Method {args.method} is not supported."
    raise ValueError(msg)
loss = pde_util.loss_mse()


@jax.jit
@jax.value_and_grad
def loss_value_and_grad(p, y):
    approx = approx_solve(y0, p)
    return loss(approx, targets=y)


# Optimize
variables_after = variables_before
opt_state = optimizer.init(variables_after)

progressbar = tqdm.tqdm(range(args.num_epochs))
progressbar.set_description("Loss ")
for _ in progressbar:
    loss, grad = loss_value_and_grad(variables_after, target_y1)
    updates, opt_state = optimizer.update(grad, opt_state)
    variables_after = optax.apply_updates(variables_after, updates)
    progressbar.set_description(f"Loss {loss:.1e}")


# Plot the solution

layout = onp.asarray(
    [
        ["truth_scale", "truth_t0", "truth_t1"],
        ["before_scale", "before_t0", "before_t1"],
        ["after_scale", "after_t0", "after_t1"],
    ]
)
figsize = (onp.shape(layout)[1] * 3, onp.shape(layout)[0] * 2)
fig, axes = plt.subplot_mosaic(layout, figsize=figsize, sharex=True, sharey=True)


def plot_t0(ax, x, /):
    kwargs_t0 = {"cmap": "Greys"}
    args_plot = x

    clr = ax.contourf(mesh[0], mesh[1], args_plot, **kwargs_t0)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_t1(ax, x, /):
    kwargs_t1 = {"cmap": "Oranges"}
    args_plot = x

    clr = ax.contourf(mesh[0], mesh[1], args_plot, **kwargs_t1)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_scale(ax, x, /):
    kwargs_scale = {"cmap": "Blues"}
    args_plot = constrain(x)
    clr = ax.contourf(mesh[0], mesh[1], args_plot, **kwargs_scale)
    fig.colorbar(clr, ax=ax)
    return ax


axes["truth_t0"].set_title(f"$y(t={pde_t0})$ (known)", fontsize="medium")
axes["truth_t1"].set_title(f"$y(t={pde_t1})$ (target)", fontsize="medium")
axes["truth_scale"].set_title("GRF / MLP (unknown)", fontsize="medium")

axes["truth_scale"].set_ylabel("Truth (GRF)")
plot_t0(axes["truth_t0"], y0[0])
plot_t1(axes["truth_t1"], target_y1[0])
plot_scale(axes["truth_scale"], grf_scale)


axes["before_scale"].set_ylabel("Before optim. (MLP)")
mlp_scale = mlp_apply(mlp_unflatten(variables_before), mesh)
approx_y1 = approx_solve(y0, variables_before)
plot_t0(axes["before_t0"], y0[0])
plot_t1(axes["before_t1"], approx_y1[0])
plot_scale(axes["before_scale"], mlp_scale)


axes["after_scale"].set_ylabel("After optim. (MLP)")
mlp_scale = mlp_apply(mlp_unflatten(variables_after), mesh)
approx_y1 = approx_solve(y0, variables_after)
plot_t0(axes["after_t0"], y0[0])
plot_t1(axes["after_t1"], approx_y1[0])
plot_scale(axes["after_scale"], mlp_scale)

plt.savefig(f"{directory_fig}/figure.pdf")
plt.show()
