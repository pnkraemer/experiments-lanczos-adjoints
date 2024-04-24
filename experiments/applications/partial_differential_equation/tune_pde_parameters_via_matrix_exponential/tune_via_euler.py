import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import numpy as onp  # for matplotlib manipulations  # noqa: ICN001
from matfree_extensions.util import exp_util, gp_util, pde_util

# Set parameters
pde_t0, pde_t1 = 0.0, 1.0
dx_time, dx_space = 2e-2, 2e-2
seed = 1

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
grf_K = gp_util.gram_matrix(grf_kernel(**grf_params))(grf_xs, grf_xs)
grf_cholesky = jnp.linalg.cholesky(grf_K + 1e-5 * jnp.eye(len(grf_K)))
key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_drift = (grf_cholesky @ grf_eps).reshape(mesh[0].shape)


# Initial condition
pde_init, _params_init = pde_util.pde_init_sine()


def init(x, p):
    return pde_init(**p)(x)


# Discretise the PDE dynamics with method of lines (MOL)
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_heat_affine(
    0.02, grf_drift, stencil=stencil, boundary=boundary
)


def vector_field(x, p):
    return pde_rhs(**p)(x)


# Prepare the solver
_mesh, unflatten_x = jax.flatten_util.ravel_pytree(mesh)


# Create the data/targets
target_p_init = {"scale_sin": 5.0, "scale_cos": 3.0}
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


mlp_init, mlp_apply = pde_util.model_mlp(mesh, activation=jnp.tanh)
key, subkey = jax.random.split(key, num=2)
mlp_params, mlp_unflatten = mlp_init(subkey)


def vector_field_mlp(x, p):
    """Evaluate parametrised PDE dynamics."""
    drift_predicted = mlp_apply(mlp_unflatten(p), mesh)
    return pde_rhs(drift=drift_predicted)(x)


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


@jax.jit
@jax.value_and_grad
def loss_function(p, x, y):
    loss = pde_util.loss_mse()
    (_, approx), _ = approx_model(p, x)
    return loss(approx, targets=y)


print(loss_function(approx_params, mesh, target_y1))

# Print the solution

layout = onp.asarray(
    [
        ["truth_t0", "truth_t1", "truth_drift"],
        ["before_t0", "before_t1", "before_drift"],
    ]
)
figsize = (onp.shape(layout)[1] * 3, onp.shape(layout)[0] * 2)
_fig, axes = plt.subplot_mosaic(layout, figsize=figsize, sharex=True, sharey=True)

axes["truth_t0"].set_title("t0")
axes["truth_t1"].set_title("t1")
axes["truth_drift"].set_title("Drift")

axes["truth_t0"].set_ylabel("Truth")
axes["truth_t0"].contourf(mesh[0], mesh[1], targets_all[0])
axes["truth_t1"].contourf(mesh[0], mesh[1], targets_all[-1])
axes["truth_drift"].contourf(mesh[0], mesh[1], grf_drift)

axes["before_t0"].set_ylabel("Before optimisation")
mlp_drift = mlp_apply(mlp_unflatten(mlp_params), mesh)
_, approx_all = approx_model(approx_params, mesh)
axes["before_t0"].contourf(mesh[0], mesh[1], approx_all[0])
axes["before_t1"].contourf(mesh[0], mesh[1], approx_all[-1])
axes["before_drift"].contourf(mesh[0], mesh[1], mlp_drift)

plt.show()
