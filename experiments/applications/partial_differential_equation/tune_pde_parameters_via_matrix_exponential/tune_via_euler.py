import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import numpy as onp  # for matplotlib manipulations  # noqa: ICN001
from matfree_extensions.util import exp_util, gp_util, pde_util

# Set parameters
t0, t1 = 0.0, 1.0
dt, dx = 2e-2, 2e-2
num_epochs = 25
plot_dt = 0.2
seed = 1

# Key
key = jax.random.PRNGKey(seed)

# Discretise the space-domain
xs_1d = jnp.arange(0.0, 1.0 + dx, step=dx)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)
print(f"Number of points: {mesh.size // 2}")

# Sample a Gaussian random field as a "true" drift
xs = mesh.reshape((2, -1)).T
kernel, kparams = gp_util.kernel_scaled_matern_32(shape_in=(2,), shape_out=())
K = gp_util.gram_matrix(kernel(**kparams))(xs, xs)
cholesky = jnp.linalg.cholesky(K + 1e-5 * jnp.eye(len(K)))
key, subkey = jax.random.split(key, num=2)
eps = jax.random.normal(subkey, shape=xs[:, 0].shape)
drift = (cholesky @ eps).reshape(mesh[0].shape)


# Initial condition
pde_init, _params_init = pde_util.pde_init_sine()


# PDE dynamics
stencil = pde_util.stencil_laplacian(dx)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_rhs_heat_affine(
    0.02, drift, stencil=stencil, boundary=boundary
)


mlp_init, mlp_apply = pde_util.model_mlp(mesh, activation=jnp.tanh)
key, subkey = jax.random.split(key, num=2)
mlp_params, mlp_unflatten = mlp_init(subkey)


def pde_rhs_p(x, p):
    """Evaluate parametrised PDE dynamics."""
    return pde_rhs(drift=mlp_apply(mlp_unflatten(p), mesh))(x)


# Prepare the solver
num_steps = int((t1 - t0) / dt)
ts = jnp.linspace(t0, t1, num=num_steps, endpoint=True)
_mesh, unflatten_x = jax.flatten_util.ravel_pytree(mesh)


# Create the data/targets
params_init = {"scale_sin": 5.0, "scale_cos": 3.0}
params_rhs = {"drift": drift}
params_true, unflatten_p = jax.flatten_util.ravel_pytree((params_init, params_rhs))
solve = pde_util.solver_euler_fixed_step(ts, lambda x, p: pde_rhs(**p)(x))
model = pde_util.model_pde(
    unflatten=(unflatten_p, unflatten_x),
    init=lambda x, s: pde_init(**s)(x),
    solve=solve,
)
(_, targets), targets_all = model(params_true, mesh)

# Sample initial parameters
key, subkey = jax.random.split(key, num=2)
params_rhs = exp_util.tree_random_like(subkey, params_rhs)
key, subkey = jax.random.split(key, num=2)
params_init = exp_util.tree_random_like(subkey, params_init)
params, unflatten_p = jax.flatten_util.ravel_pytree((params_init, mlp_params))

# Build a model
solve = pde_util.solver_euler_fixed_step(ts, pde_rhs_p)
model = pde_util.model_pde(
    unflatten=(unflatten_p, unflatten_x),
    init=lambda x, s: pde_init(**s)(x),
    solve=solve,
)


# Evaluate the loss
loss = pde_util.loss_mse()
(_t1, before_y1), before_all = model(params, mesh)
print(loss(before_y1, targets=targets))


# Print the solution

layout = onp.asarray([["truth_t0", "truth_t1"], ["before_t0", "before_t1"]])
figsize = (onp.shape(layout)[1] * 3, onp.shape(layout)[0] * 2)
fig, axes = plt.subplot_mosaic(layout, figsize=figsize, sharex=True, sharey=True)

axes["truth_t0"].contourf(mesh[0], mesh[1], targets_all[0])
axes["truth_t1"].contourf(mesh[0], mesh[1], targets_all[-1])

axes["before_t0"].contourf(mesh[0], mesh[1], before_all[0])
axes["before_t1"].contourf(mesh[0], mesh[1], before_all[-1])

plt.show()
