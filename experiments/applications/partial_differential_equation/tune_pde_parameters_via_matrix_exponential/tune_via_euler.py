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
# todo: simplify this script a little bit
# todo: quantify the reconstruction errors a little bit
# todo: run for different solvers
# todo: figure out what to plot...

# Set parameters
pde_t0, pde_t1 = 0.0, 0.05
dx_time, dx_space = 0.005, 0.02
seed = 1
train_num_epochs = 1000
train_display_every = 10
mlp_features = [20, 20, 1]
mlp_activation = jax.nn.tanh
optimizer = optax.adam(1e-2)
arnoldi_depth = 20
print(arnoldi_depth)

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

key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_init = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)

# Initial condition
pde_init, params_init = pde_util.pde_init_bell(7.5)


def init(x, _p):
    u0 = grf_init
    return jnp.stack([u0, u0])


y0 = init(mesh, ())


# Discretise the PDE dynamics with method of lines (MOL)
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_wave_anisotropic(
    grf_scale, constrain=constrain, stencil=stencil, boundary=boundary
)


def vector_field(x, p):
    return pde_rhs(**p)(x)


# Prepare the solver
_mesh, unflatten_x = jax.flatten_util.ravel_pytree(mesh)


# Create the data/targets
target_p_init = {}
target_p_rhs = {"scale": grf_scale}
target_params, unflatten_p = jax.flatten_util.ravel_pytree(
    (target_p_init, target_p_rhs)
)
target_solve = pde_util.solver_euler_fixed_step(solve_ts, vector_field)
target_model = pde_util.model_pde(
    unflatten=(unflatten_p, unflatten_x), init=init, solve=target_solve
)
(_, target_y1), targets_all = target_model(target_params, mesh)


# Sample initial parameters


mlp_init, mlp_apply = pde_util.model_mlp(mesh, mlp_features, activation=mlp_activation)
key, subkey = jax.random.split(key, num=2)
mlp_params, mlp_unflatten = mlp_init(subkey)


def vector_field_mlp(x, p):
    """Evaluate parametrised PDE dynamics."""
    scale_predicted = mlp_apply(mlp_unflatten(p), mesh)
    scale = scale_predicted
    return pde_rhs(scale=scale)(x)


key, subkey = jax.random.split(key, num=2)
approx_p_init = exp_util.tree_random_like(subkey, target_p_init)
approx_p_rhs = mlp_params
approx_p_tree = (approx_p_init, approx_p_rhs)
approx_params, unflatten_p = jax.flatten_util.ravel_pytree(approx_p_tree)

# Build a model
# expm = pde_util.expm_arnoldi(arnoldi_depth)
# approx_solve = pde_util.solver_arnoldi(pde_t0, pde_t1, vector_field_mlp, expm=expm)
# approx_solve = pde_util.solver_euler_fixed_step(solve_ts, vector_field_mlp)
approx_solve = pde_util.solver_diffrax(
    pde_t0, pde_t1, vector_field_mlp, method="dopri5"
)
approx_model = pde_util.model_pde(
    unflatten=(unflatten_p, unflatten_x), init=init, solve=approx_solve
)
loss = pde_util.loss_mse()


@jax.jit
@jax.value_and_grad
def loss_value_and_grad(p, x, y):
    (_, approx), _ = approx_model(p, x)
    return loss(approx, targets=y)


# Optimize

variables = approx_params
opt_state = optimizer.init(variables)

for epoch in range(train_num_epochs):
    loss, grad = loss_value_and_grad(variables, mesh, target_y1)
    updates, opt_state = optimizer.update(grad, opt_state)
    variables = optax.apply_updates(variables, updates)
    if epoch % train_display_every == 0:
        print(epoch, loss)


# Print the solution

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
    clr = ax.contourf(mesh[0], mesh[1], args_plot[0], **kwargs_t0)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_t1(ax, args, /):
    kwargs_t1 = {"cmap": "Oranges_r"}

    args_plot = args
    clr = ax.contourf(mesh[0], mesh[1], args_plot[0], **kwargs_t1)
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
plot_t0(axes["truth_t0"], y0)
plot_t1(axes["truth_t1"], targets_all[-1])
plot_scale(axes["truth_scale"], grf_scale)


axes["before_scale"].set_ylabel("Before optim. (MLP)")
mlp_scale = mlp_apply(mlp_unflatten(mlp_params), mesh)
(_, approx_y1), _approx_all = approx_model(approx_params, mesh)
plot_t0(axes["before_t0"], y0)
plot_t1(axes["before_t1"], approx_y1)
plot_scale(axes["before_scale"], mlp_scale)


axes["after_scale"].set_ylabel("After optim. (MLP)")
mlp_params = unflatten_p(variables)[1]
mlp_scale = mlp_apply(mlp_unflatten(mlp_params), mesh)
(_, approx_y1), _approx_all = approx_model(variables, mesh)
plot_t0(axes["after_t0"], y0)
plot_t1(axes["after_t1"], approx_y1)
plot_scale(axes["after_scale"], mlp_scale)


plt.show()
