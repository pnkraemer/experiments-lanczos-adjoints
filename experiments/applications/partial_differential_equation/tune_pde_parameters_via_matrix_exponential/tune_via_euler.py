import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
from matfree_extensions.util import exp_util, pde_util

# Set parameters
t0, t1 = 0.0, 1.0
dt, dx = 1e-1, 1e-1
num_epochs = 25
plot_dt = 0.2
seed = 1

# Key
key = jax.random.PRNGKey(seed)

# Discretise the space-domain
xs_1d = jnp.arange(0.0, 1.0 + dx, step=dx)
mesh = pde_util.mesh_2d_tensorproduct(xs_1d, xs_1d)
print(f"Number of points: {mesh.size // 2}")

# Initial condition
pde_init, params_init = pde_util.pde_2d_init_bell()
key, subkey = jax.random.split(key, num=2)
params_init = exp_util.tree_random_like(subkey, params_init)

# PDE dynamics
stencil = pde_util.stencil_2d_laplacian(dx)
boundary = pde_util.boundary_neumann()
pde_rhs, params_rhs = pde_util.pde_2d_rhs_laplacian(stencil=stencil, boundary=boundary)
key, subkey = jax.random.split(key, num=2)
params_rhs = exp_util.tree_random_like(subkey, params_rhs)

# Print a test-run of results
num_steps = int((t1 - t0) / dt)
ts = jnp.linspace(t0, t1, num=num_steps, endpoint=True)
solve = pde_util.solver_euler_fixed_step(ts, lambda x, p: pde_rhs(**p)(x))
y0 = pde_init(**params_init)(mesh)
y_final, ys = solve(y0, params_rhs)

# Build a model
params, unflatten_p = jax.flatten_util.ravel_pytree((params_init, params_rhs))
model = pde_util.model_pde(
    unflatten=unflatten_p, init=lambda x, s: pde_init(**s)(x), solve=solve
)
y1, y_all = model(params, mesh)

print(y1)
