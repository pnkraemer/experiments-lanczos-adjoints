import argparse
import os
import time
import warnings

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import optax
import tqdm
from matfree_extensions.util import exp_util, gp_util, pde_util

# todo: save all reconstruction errors (y1, scale, fwd_raw, bwd_raw) in a file
# todo: run for different solvers
# todo: verify somehow that we do solve a wave equation!
# todo: plot the RHS matrix to see whether it is indeed symmetric


# Make directories
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
optimizer = optax.adam(1e-2)

# Process the parameters
key = jax.random.PRNGKey(args.seed)
solve_ts = jnp.linspace(pde_t0, pde_t1, num=args.num_matvecs, endpoint=True)
arnoldi_depth = args.num_matvecs


# Discretise the space-domain
xs_1d = jnp.linspace(0.0, 1.0, endpoint=True, num=args.num_dx_points)
dx_space = jnp.diff(xs_1d)[0]
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)


def constrain(arg):
    """Constrain the PDE-scale to strictly positive values."""
    return 0.1 * arg**2


# Sample a Gaussian random field as a "true" scale
grf_xs = mesh.reshape((2, -1)).T
grf_kernel, _ = gp_util.kernel_scaled_rbf(shape_in=(2,), shape_out=())
kernel_fun = grf_kernel(raw_lengthscale=-0.75, raw_outputscale=-4.0)
grf_K = gp_util.gram_matrix(kernel_fun)(grf_xs, grf_xs)

# Sample  # todo: sample with Lanczos? Otherwise we go crazy here...
(w, v) = jnp.linalg.eigh(grf_K)
w = jnp.maximum(0.0, w)  # clamp to allow sqrts
grf_factor = v * jnp.sqrt(w[..., None, :])

key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_scale = (grf_factor @ grf_eps).reshape(mesh[0].shape)


# Initial condition
key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_init = (grf_factor @ grf_eps).reshape(mesh[0].shape)

key, subkey = jax.random.split(key, num=2)
grf_eps = jax.random.normal(subkey, shape=grf_xs[:, 0].shape)
grf_init_diff = (grf_factor @ grf_eps).reshape(mesh[0].shape)
y0 = jnp.stack([grf_init, grf_init_diff])
print(f"\nNumber of points: {mesh.size // 2}")
print(f"Number of ODE dimensions: {y0.size}\n")


# Discretise the PDE dynamics with method of lines (MOL)
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_wave_anisotropic(
    grf_scale, constrain=constrain, stencil=stencil, boundary=boundary
)


# Create the data/targets


@jax.jit
def vector_field(x, p):
    """Evaluate the PDE dynamics."""
    return pde_rhs(scale=p)(x)


# Time the vector field (count how many iterations we can fit in a second)
vector_field(y0, grf_scale).block_until_ready()  # pre compile
t0 = time.perf_counter()
ct = 0
while (vf_time := (time.perf_counter() - t0)) < 1.0:
    vector_field(y0, grf_scale).block_until_ready()
    ct += 1
vf_time /= ct
print(f"\nRHS evaluation: {vf_time} seconds")
# x2 because we assume that the adjoint pass also evaluates the RHS
print(f"Projected runtime per iteration: ~{vf_time * args.num_matvecs * 2} seconds\n")

solve_ts_data = jnp.linspace(pde_t0, pde_t1, endpoint=True, num=10_000)
target_solve = pde_util.solver_euler_fixed_step(solve_ts_data, vector_field)

# Build an approximate model
if args.method == "arnoldi":
    expm = pde_util.expm_arnoldi(arnoldi_depth)
    approx_solve = pde_util.solver_expm(pde_t0, pde_t1, vector_field, expm=expm)
elif args.method == "expm-pade":
    if args.num_dx_points > 20:
        msg = f"Careful: method '{args.method}' is very expensive"
        warnings.warn(msg, stacklevel=1)
    expm = pde_util.expm_pade()
    approx_solve = pde_util.solver_expm(pde_t0, pde_t1, vector_field, expm=expm)
elif args.method == "euler":
    approx_solve = pde_util.solver_euler_fixed_step(solve_ts, vector_field)
elif args.method == "diffrax-tsit5":
    adjoint = "recursive_checkpoint"
    approx_solve = pde_util.solver_diffrax(
        pde_t0,
        pde_t1,
        vector_field,
        num_steps=args.num_matvecs // 5,
        method="tsit5",
        adjoint=adjoint,
    )
elif args.method == "diffrax-euler":
    adjoint = "recursive_checkpoint"

    approx_solve = pde_util.solver_diffrax(
        pde_t0,
        pde_t1,
        vector_field,
        num_steps=args.num_matvecs,
        method="euler",
        adjoint=adjoint,
    )
elif args.method == "diffrax-heun":
    adjoint = "recursive_checkpoint"
    approx_solve = pde_util.solver_diffrax(
        pde_t0,
        pde_t1,
        vector_field,
        num_steps=args.num_matvecs // 2,
        method="heun",
        adjoint=adjoint,
    )
else:
    msg = f"Method {args.method} is not supported."
    raise ValueError(msg)

target_y1 = target_solve(y0, grf_scale)
approx_y1 = approx_solve(y0, grf_scale)

fwd_error = jnp.sqrt(jnp.mean((approx_y1 - target_y1) ** 2))
print("\nForward error:", fwd_error)

key, subkey = jax.random.split(key, num=2)
u = jax.random.normal(subkey, shape=y0.shape)
target_jacrev = jax.grad(lambda z: jnp.vdot(u, target_solve(y0, z)))(grf_scale)
approx_jacrev = jax.grad(lambda z: jnp.vdot(u, approx_solve(y0, z)))(grf_scale)

bwd_error = jnp.sqrt(jnp.mean((approx_jacrev - target_jacrev) ** 2))
print("Backward error:", bwd_error, "\n")


# Set up parameter approximation
mlp_init, mlp_apply = pde_util.model_mlp(mesh, mlp_features, activation=mlp_activation)
key, mlp_key = jax.random.split(key, num=2)
variables_before, mlp_unflatten = mlp_init(subkey)
print(f"\nNumber of parameters: {variables_before.size}\n")


# Create a loss function
loss = pde_util.loss_mse()


@jax.jit
@jax.value_and_grad
def loss_value_and_grad(p, y):
    scale = mlp_apply(mlp_unflatten(p), mesh)
    approx = approx_solve(y0, scale)
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
    progressbar.set_description(f"Loss {loss:.3e}")


# Save to a file
scale_before = mlp_apply(mlp_unflatten(variables_before), mesh)
scale_after = mlp_apply(mlp_unflatten(variables_after), mesh)
y1_before = approx_solve(y0, scale_before)
y1_after = approx_solve(y0, scale_after)
jnp.save(f"{directory_results}{args.method}_y0.npy", y0)
jnp.save(
    f"{directory_results}{args.method}_scale_mlp_before.npy", constrain(scale_before)
)
jnp.save(
    f"{directory_results}{args.method}_scale_mlp_after.npy", constrain(scale_after)
)
jnp.save(f"{directory_results}{args.method}_scale_grf.npy", constrain(grf_scale))
jnp.save(f"{directory_results}{args.method}_y1_target.npy", target_y1)
jnp.save(f"{directory_results}{args.method}_y1_approx_before.npy", y1_before)
jnp.save(f"{directory_results}{args.method}_y1_approx_after.npy", y1_after)
