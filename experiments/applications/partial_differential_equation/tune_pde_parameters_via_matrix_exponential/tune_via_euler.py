import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import numpy as onp  # for matplotlib manipulations  # noqa: ICN001
import optax
from matfree_extensions.util import exp_util, gp_util, pde_util

# Set parameters
pde_t0, pde_t1 = 0.0, 0.05
dx_time, dx_space = 0.005, 0.0125
seed = 1
train_num_epochs = 1000
mlp_features = [20, 20, 1]
mlp_activation = jax.nn.tanh
optimizer = optax.adam(1e-3)


# Process the parameters
key = jax.random.PRNGKey(seed)
solve_num_steps = int((pde_t1 - pde_t0) / dx_time)
solve_ts = jnp.linspace(pde_t0, pde_t1, num=solve_num_steps, endpoint=True)

# Discretise the space-domain
xs_1d = jnp.arange(0.0, 1.0 + dx_space, step=dx_space)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)
print(f"Number of points: {mesh.size // 2}")


# Sample a Gaussian random field as a "true" drift
grf_xs = mesh.reshape((2, -1)).T
grf_matern = gp_util.kernel_scaled_matern_32(shape_in=(2,), shape_out=())
grf_kernel, grf_params = grf_matern
key, subkey = jax.random.split(key, num=2)
grf_params = exp_util.tree_random_like(subkey, grf_params)

grf_K = gp_util.gram_matrix(grf_kernel(**grf_params))(grf_xs, grf_xs)
grf_cholesky = jnp.linalg.cholesky(grf_K + 2e-5 * jnp.eye(len(grf_K)))
key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_drift = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)

key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_init = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)

# Initial condition
pde_init, params_init = pde_util.pde_init_bell(7.5)


def init(x, _p):
    # u0 =  pde_init(**params_init)(x)
    u0 = grf_init
    return jnp.stack([u0, u0])


y0 = init(mesh, ())


# Discretise the PDE dynamics with method of lines (MOL)
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_wave_affine(
    grf_drift, stencil=stencil, boundary=boundary
)


def vector_field(x, p):
    return pde_rhs(**p)(x)


# Prepare the solver
_mesh, unflatten_x = jax.flatten_util.ravel_pytree(mesh)


# Create the data/targets
target_p_init = {}
target_p_rhs = {"drift": grf_drift}
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
    drift_predicted = mlp_apply(mlp_unflatten(p), mesh)
    drift = drift_predicted
    return pde_rhs(drift=drift)(x)


key, subkey = jax.random.split(key, num=2)
approx_p_init = exp_util.tree_random_like(subkey, target_p_init)
approx_p_rhs = mlp_params
approx_p_tree = (approx_p_init, approx_p_rhs)
approx_params, unflatten_p = jax.flatten_util.ravel_pytree(approx_p_tree)

# Build a model
approx_solve = pde_util.solver_euler_fixed_step(solve_ts, vector_field_mlp)
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
    # Subsample data
    key, subkey = jax.random.split(key)

    # Apply an optimizer-step
    loss, grad = loss_value_and_grad(variables, mesh, target_y1)
    updates, opt_state = optimizer.update(grad, opt_state)
    variables = optax.apply_updates(variables, updates)
    print(epoch, loss)


# Print the solution

layout = onp.asarray(
    [
        ["truth_drift", "truth_t0", "truth_t1"],
        ["before_drift", "before_t0", "before_t1"],
        ["after_drift", "after_t0", "after_t1"],
    ]
)
figsize = (onp.shape(layout)[1] * 3, onp.shape(layout)[0] * 2)
fig, axes = plt.subplot_mosaic(layout, figsize=figsize, sharex=True, sharey=True)


def plot_t0(ax, args, /):
    kwargs_t0 = {
        # "vmin": jnp.amin(targets_all[0]),
        # "vmax": jnp.amax(targets_all[0]),
        "cmap": "Greys_r"
    }
    args_plot = args + jnp.finfo(jnp.dtype(args)).eps
    clr = ax.contourf(mesh[0], mesh[1], args_plot[0], **kwargs_t0)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_t1(ax, args, /):
    kwargs_t1 = {
        # "vmin": jnp.amin(targets_all[-1]),
        # "vmax": jnp.amax(targets_all[-1]),
        "cmap": "Oranges_r"
    }

    args_plot = args + jnp.finfo(jnp.dtype(args)).eps
    clr = ax.contourf(mesh[0], mesh[1], args_plot[0], **kwargs_t1)
    fig.colorbar(clr, ax=ax)
    return ax


def plot_drift(ax, args, /):
    kwargs_drift = {
        # "vmin": jnp.amin(grf_drift),
        # "vmax": jnp.amax(grf_drift),
        "cmap": "Blues"
    }
    args_plot = 0.001 + args**2

    clr = ax.contourf(mesh[0], mesh[1], args_plot, **kwargs_drift)
    fig.colorbar(clr, ax=ax)
    return ax


axes["truth_t0"].set_title("$y(t=0)$ (known)", fontsize="medium")
axes["truth_t1"].set_title("$y(t=1)$ (target)", fontsize="medium")
axes["truth_drift"].set_title("GRF / MLP (unknown)", fontsize="medium")

axes["truth_drift"].set_ylabel("Truth (GRF)")
plot_t0(axes["truth_t0"], y0)
plot_t1(axes["truth_t1"], targets_all[-1])
plot_drift(axes["truth_drift"], grf_drift)


axes["before_drift"].set_ylabel("Before optim. (MLP)")
mlp_drift = mlp_apply(mlp_unflatten(mlp_params), mesh)
_, approx_all = approx_model(approx_params, mesh)
plot_t0(axes["before_t0"], y0)
plot_t1(axes["before_t1"], approx_all[-1])
plot_drift(axes["before_drift"], mlp_drift)


axes["after_drift"].set_ylabel("After optim. (MLP)")
mlp_params = unflatten_p(variables)[1]
mlp_drift = mlp_apply(mlp_unflatten(mlp_params), mesh)
_, approx_all = approx_model(variables, mesh)
plot_t0(axes["after_t0"], y0)
plot_t1(axes["after_t1"], approx_all[-1])
plot_drift(axes["after_drift"], mlp_drift)


plt.show()
