import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import numpy as onp  # for matplotlib manipulations  # noqa: ICN001
import optax
from matfree_extensions.util import exp_util, gp_util, pde_util

# todo: add a "naive" matrix exponential solver
# todo: expose some of the solver options via argparse
# todo: quantify the reconstruction errors a little bit
# todo: run for different solvers
# todo: figure out what to plot...

# Set parameters
pde_t0, pde_t1 = 0.0, 0.05
dx_time, dx_space = 0.005, 0.02
seed = 1
train_num_epochs = 1500
train_display_every = 10
mlp_features = [20, 20, 1]
mlp_activation = jax.nn.tanh
optimizer = optax.adam(1e-2)
arnoldi_depth = 20

# Process the parameters
key = jax.random.PRNGKey(seed)
solve_num_steps = int((pde_t1 - pde_t0) / dx_time)
solve_ts = jnp.linspace(pde_t0, pde_t1, num=solve_num_steps, endpoint=True)

# Discretise the space-domain
xs_1d = jnp.arange(0.0, 1.0 + dx_space, step=dx_space)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)
print(f"Number of points: {mesh.size // 2}")


def constrain(arg):
    """Constrain the PDE-scale to strictly positive values."""
    return 0.001 + arg**2


# Sample a Gaussian random field as a "true" scale
grf_xs = mesh.reshape((2, -1)).T
grf_matern = gp_util.kernel_scaled_matern_32(shape_in=(2,), shape_out=())
grf_kernel, grf_params = grf_matern
key, subkey = jax.random.split(key, num=2)
grf_params = exp_util.tree_random_like(subkey, grf_params)

grf_K = gp_util.gram_matrix(grf_kernel(**grf_params))(grf_xs, grf_xs)
grf_cholesky = jnp.linalg.cholesky(grf_K + 2e-5 * jnp.eye(len(grf_K)))
key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_scale = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)


# Initial condition
key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_init = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)
y0 = jnp.stack([grf_init, grf_init])


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


target_solve = pde_util.solver_euler_fixed_step(solve_ts, vector_field)
target_y1 = target_solve(y0)

# Build an approximate model
mlp_init, mlp_apply = pde_util.model_mlp(mesh, mlp_features, activation=mlp_activation)
key, subkey = jax.random.split(key, num=2)
variables_before, mlp_unflatten = mlp_init(subkey)


def vector_field_mlp(x, p):
    """Evaluate the MLP-parametrised PDE dynamics."""
    scale = mlp_apply(mlp_unflatten(p), mesh)
    return pde_rhs(scale=scale)(x)


# Create a loss function

approx_solve = pde_util.solver_euler_fixed_step(solve_ts, vector_field_mlp)
loss = pde_util.loss_mse()


@jax.jit
@jax.value_and_grad
def loss_value_and_grad(p, y):
    approx = approx_solve(y0, p)
    return loss(approx, targets=y)


# Optimize
variables_after = variables_before
opt_state = optimizer.init(variables_after)


for epoch in range(train_num_epochs):
    loss, grad = loss_value_and_grad(variables_after, target_y1)
    updates, opt_state = optimizer.update(grad, opt_state)
    variables_after = optax.apply_updates(variables_after, updates)
    if epoch % train_display_every == 0:
        print(epoch, loss)


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


def plot_t0(ax, args, /):
    kwargs_t0 = {"cmap": "Greys_r"}
    args_plot = args

    clr = ax.contourf(mesh[0], mesh[1], args_plot, **kwargs_t0)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_t1(ax, args, /):
    kwargs_t1 = {"cmap": "Oranges_r"}
    args_plot = args

    clr = ax.contourf(mesh[0], mesh[1], args_plot, **kwargs_t1)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_scale(ax, args, /):
    kwargs_scale = {"cmap": "Blues"}
    args_plot = constrain(args)
    clr = ax.contourf(mesh[0], mesh[1], args_plot, **kwargs_scale)
    fig.colorbar(clr, ax=ax)
    return ax


axes["truth_t0"].set_title("$y(t=0)$ (known)", fontsize="medium")
axes["truth_t1"].set_title("$y(t=1)$ (target)", fontsize="medium")
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


plt.show()
