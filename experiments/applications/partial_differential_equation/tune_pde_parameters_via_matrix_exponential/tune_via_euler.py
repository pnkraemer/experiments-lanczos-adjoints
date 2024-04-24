import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
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

# Initial condition
# pde_init, params_init = pde_util.pde_init_bell(7.0)
pde_init, params_init = pde_util.pde_init_sine()
key, subkey = jax.random.split(key, num=2)
params_init = exp_util.tree_random_like(subkey, params_init)


# Sample a Gaussian random field as a "true" drift
xs = mesh.reshape((2, -1))
kernel, params = gp_util.kernel_scaled_matern_32(shape_in=(2,), shape_out=())
K = gp_util.gram_matrix(kernel(**params))(xs.T, xs.T)
cholesky = jnp.linalg.cholesky(K + 1e-5 * jnp.eye(len(K)))
key, subkey = jax.random.split(key, num=2)
eps = jax.random.normal(subkey, shape=xs[0].shape)
drift = (cholesky @ eps).reshape(mesh[0].shape)


# PDE dynamics
stencil = pde_util.stencil_laplacian(dx)
boundary = pde_util.boundary_neumann()
pde_rhs, params_rhs = pde_util.pde_rhs_heat_affine(
    0.02, drift, stencil=stencil, boundary=boundary
)
key, subkey = jax.random.split(key, num=2)
params_rhs = exp_util.tree_random_like(subkey, params_rhs)
params_rhs["drift"] = drift

# Print a test-run of results
num_steps = int((t1 - t0) / dt)
ts = jnp.linspace(t0, t1, num=num_steps, endpoint=True)
solve = pde_util.solver_euler_fixed_step(ts, lambda x, p: pde_rhs(**p)(x))
y0 = pde_init(**params_init)(mesh)
(_t, targets), _ys = solve(y0, params_rhs)


# Build a model
params, unflatten_p = jax.flatten_util.ravel_pytree((params_init, params_rhs))
_mesh, unflatten_x = jax.flatten_util.ravel_pytree(mesh)
model = pde_util.model_pde(
    unflatten=(unflatten_p, unflatten_x),
    init=lambda x, s: pde_init(**s)(x),
    solve=solve,
)

(t1, y1), y_all = model(params, mesh)

# Evaluate the loss
loss = pde_util.loss_mse()
print(loss(model(params, mesh)[0][1], targets=targets))


# Print the solution
fig, axes = plt.subplot_mosaic([["t0", "t1"]], figsize=(8, 3), sharex=True, sharey=True)
axes["t0"].contourf(mesh[0], mesh[1], y0)
img = axes["t1"].contourf(mesh[0], mesh[1], y1)
plt.colorbar(img)
plt.show()
