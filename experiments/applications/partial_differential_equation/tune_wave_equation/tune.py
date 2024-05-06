import argparse
import os
import pickle
import time
import warnings

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import optax
import tqdm
from matfree_extensions.util import exp_util, gp_util, pde_util

# todo: verify somehow that we do solve the PDE!
# todo: turn the sampler into a Lanczos sampler (so we can scale!)
# todo: 3d?


# Make directories
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--method", required=True)
parser.add_argument("--num_matvecs", type=int, required=True)
parser.add_argument("--num_dx_points", type=int, required=True)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--plot_matrix", action="store_true")
parser.add_argument("--x64", action="store_true")
args = parser.parse_args()
print(args)

jax.config.update("jax_enable_x64", args.x64)


# Set parameters
pde_t0, pde_t1 = 0.0, 1.0
mlp_features = [20, 20, 1]
mlp_activation = jax.nn.tanh
optimizer = optax.adam(2e-2)  # weirdly important when matrices get large

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
    return arg**2


# Sample a Gaussian random field as a "true" scale
grf_xs = mesh.reshape((2, -1)).T
grf_kernel, _ = gp_util.kernel_scaled_rbf(shape_in=(2,), shape_out=())
kernel_fun = grf_kernel(raw_lengthscale=-0.75, raw_outputscale=-5.0)
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


if args.plot_matrix:
    y0flat, unflat = jax.flatten_util.ravel_pytree(y0)
    matrix = jax.jacfwd(
        lambda g: jax.flatten_util.ravel_pytree(vector_field(unflat(g), grf_scale))[0]
    )(y0flat)
    plt.spy(matrix)
    plt.show()


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

solve_ts_data = jnp.linspace(pde_t0, pde_t1, endpoint=True, num=20_000)

target_solve = pde_util.solver_diffrax(
    pde_t0, pde_t1, vector_field, num_steps=1000, method="dopri8", adjoint="backsolve"
)

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
    # adjoint = "recursive_checkpoint"
    adjoint = "backsolve"
    approx_solve = pde_util.solver_diffrax(
        pde_t0,
        pde_t1,
        vector_field,
        num_steps=args.num_matvecs // 5,
        method="tsit5",
        adjoint=adjoint,
    )
elif args.method == "diffrax-euler":
    # adjoint = "recursive_checkpoint"
    adjoint = "backsolve"

    approx_solve = pde_util.solver_diffrax(
        pde_t0,
        pde_t1,
        vector_field,
        num_steps=args.num_matvecs,
        method="euler",
        adjoint=adjoint,
    )
elif args.method == "diffrax-euler-implicit":
    # adjoint = "recursive_checkpoint"
    adjoint = "backsolve"

    approx_solve = pde_util.solver_diffrax(
        pde_t0,
        pde_t1,
        vector_field,
        num_steps=args.num_matvecs,
        method="euler-implicit",
        adjoint=adjoint,
    )
elif args.method == "diffrax-heun":
    # adjoint = "recursive_checkpoint"
    adjoint = "backsolve"
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

stats = {}
target_y1 = target_solve(y0, grf_scale)

fun = jax.jit(approx_solve)
approx_y1 = fun(y0, grf_scale).block_until_ready()
fwd_error = jnp.mean((approx_y1 - target_y1) ** 2)
stats["MSE: FWD"] = fwd_error
print("\nForward error:", fwd_error)

t0 = time.perf_counter()
for _ in range(10):
    fun(y0, grf_scale).block_until_ready()
t1 = time.perf_counter()
stats["Time: FWD"] = (t1 - t0) / 10

key, subkey = jax.random.split(key, num=2)
u = jax.random.normal(subkey, shape=y0.shape)
target_jacrev = jax.grad(lambda z: jnp.vdot(u, target_solve(y0, z)))(grf_scale)

fun = jax.jit(jax.grad(lambda z: jnp.vdot(u, approx_solve(y0, z))))
approx_jacrev = fun(grf_scale).block_until_ready()
bwd_error = jnp.mean((approx_jacrev - target_jacrev) ** 2)
stats["MSE: REV"] = bwd_error

print("Backward error:", bwd_error, "\n")
t0 = time.perf_counter()
for _ in range(10):
    fun(grf_scale).block_until_ready()
t1 = time.perf_counter()
stats["Time: REV"] = (t1 - t0) / 10


# Set up parameter approximation
mlp_init, mlp_apply = pde_util.model_mlp(
    mesh, mlp_features, activation=mlp_activation, output_scale_raw=-5.0
)
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
    try:
        loss, grad = loss_value_and_grad(variables_after, target_y1)
        updates, opt_state = optimizer.update(grad, opt_state)
        variables_after = optax.apply_updates(variables_after, updates)
        progressbar.set_description(f"Loss {loss:.3e}")
    except KeyboardInterrupt:
        break

# Save to a file
# Todo: subsample the meshes before saving?

scale_grf = constrain(grf_scale)
scale_before = constrain(mlp_apply(mlp_unflatten(variables_before), mesh))
scale_after = constrain(mlp_apply(mlp_unflatten(variables_after), mesh))
y1_before = approx_solve(y0, scale_before)
y1_after = approx_solve(y0, scale_after)
jnp.save(f"{directory_results}{args.method}_y0.npy", y0)
jnp.save(f"{directory_results}{args.method}_scale_mlp_before.npy", scale_before)
jnp.save(f"{directory_results}{args.method}_scale_mlp_after.npy", scale_after)
jnp.save(f"{directory_results}{args.method}_scale_grf.npy", scale_grf)
jnp.save(f"{directory_results}{args.method}_y1_target.npy", target_y1)
jnp.save(f"{directory_results}{args.method}_y1_approx_before.npy", y1_before)
jnp.save(f"{directory_results}{args.method}_y1_approx_after.npy", y1_after)

error_y1 = jnp.mean((y1_after - target_y1) ** 2)
error_scale = jnp.mean((scale_grf - scale_after) ** 2)
stats["MSE: Sim."] = error_y1
stats["MSE: Param."] = error_scale


stats = jax.tree_util.tree_map(float, stats)
with open(f"{directory_results}{args.method}_stats.pkl", "wb") as handle:
    pickle.dump(stats, handle)
