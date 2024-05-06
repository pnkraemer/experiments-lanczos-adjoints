"""Evaluate work vs. precision of value_and_grad of matrix exponentials."""

import argparse
import os
import time

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import tqdm
from matfree_extensions.util import exp_util, pde_util

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
parser.add_argument("--method", type=str, required=True, help="Eg. 'arnoldi'")
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--num_steps_max", type=int, default=10)
args = parser.parse_args()
print(args)

# Load data
path = f"./data/pde_wave/{args.resolution}x{args.resolution}"

inputs = jnp.load(f"{path}_data_inputs.npy")
targets = jnp.load(f"{path}_data_targets.npy")
parameter = jnp.load(f"{path}_data_parameter.npy")


# Create a mesh
xs_1d = jnp.linspace(0.0, 1.0, endpoint=True, num=args.resolution)
mesh = pde_util.mesh_tensorproduct(xs_1d, xs_1d)

# Discretise the PDE
dx_space = jnp.diff(xs_1d)[0]
stencil = pde_util.stencil_laplacian(dx_space)
boundary = pde_util.boundary_neumann()
pde_rhs, _params_rhs = pde_util.pde_wave_anisotropic(
    parameter, constrain=jnp.square, stencil=stencil, boundary=boundary
)


def vector_field(x, p):
    """Evaluate the PDE dynamics."""
    return pde_rhs(scale=p)(x)


# Solve the problem


def loss_function(solver):
    k = jax.random.PRNGKey(1421)
    u = jax.random.uniform(k, shape=mesh.shape)

    def fun(p, y0):
        """Compute the norm of the solution."""
        y1, info = solver(y0, p)
        return jnp.vdot(u, y1), info

    return jax.jit(jax.value_and_grad(fun, has_aux=True))


# Create a reference solver
kwargs = {"num_steps": 50, "method": "dopri8", "adjoint": "direct"}
solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

loss = loss_function(solve)

# Precompile
(value, aux), gradient = loss(parameter, inputs[0])
value.block_until_ready()
gradient.block_until_ready()


ts = []
for _ in range(args.num_runs):
    t0 = time.perf_counter()
    (val, aux), grad = loss(parameter, inputs[0])
    val.block_until_ready()
    grad.block_until_ready()
    t1 = time.perf_counter()
    ts.append(t1 - t0)

ts = jnp.stack(ts)

num_steps = jnp.arange(1, args.num_steps_max)
errors_fwd = []
errors_rev = []
Ns = []
ts_all = []

progressbar = tqdm.tqdm(num_steps)
for nstp in progressbar:
    nstp = int(nstp)
    if nstp > parameter.size:
        print("The Krylov depth would exceed the matrix size.")
        break

    if args.method == "arnoldi":
        expm = pde_util.expm_arnoldi(nstp)
        solve = pde_util.solver_expm(0.0, 1.0, vector_field, expm=expm)

    elif args.method == "diffrax:euler+backsolve":
        method, adjoint = "euler", "backsolve"
        kwargs = {"num_steps": nstp, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

    elif args.method == "diffrax:heun+recursive_checkpoint":
        method, adjoint = "heun", "recursive_checkpoint"
        kwargs = {"num_steps": nstp, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

    elif args.method == "diffrax:tsit5+recursive_checkpoint":
        method, adjoint = "tsit5", "recursive_checkpoint"
        kwargs = {"num_steps": nstp, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

    elif args.method == "diffrax:dopri5+backsolve":
        method, adjoint = "dopri5", "backsolve"
        kwargs = {"num_steps": nstp, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0.0, 1.0, vector_field, **kwargs)

    else:
        msg = f"The method {args.method} is not supported."
        raise ValueError(msg)

    # Compute values and gradients (and precompile while we're at it)
    loss = loss_function(solve)
    (f, aux), df = loss(parameter, inputs[0])

    # Compute the error
    nugget = jnp.finfo(targets).eps
    error = pde_util.loss_mse_relative(nugget=nugget)
    fwd = jnp.sqrt(error(f, targets=value))
    rev = jnp.sqrt(error(df, targets=gradient))
    n = aux["num_matvecs"]
    errors_fwd.append(fwd)
    errors_rev.append(rev)
    Ns.append(int(n))
    progressbar.set_description(f"(N={n}, fwd={fwd:.1e}, rev={rev:.1e}))")

    # Compute the run times
    # todo: move to a dedicated function
    ts = []
    for _ in range(args.num_runs):
        t0 = time.perf_counter()
        (val, aux), grad = loss(parameter, inputs[0])
        val.block_until_ready()
        grad.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)

    ts = jnp.stack(ts)
    ts_all.append(ts)

    # Clear caches in the hopes of avoiding memory issues?
    jax.clear_caches()

directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)

jnp.save(f"{directory}/wp_{args.method}_Ns", Ns)
jnp.save(f"{directory}/wp_{args.method}_ts", jnp.asarray(ts_all))
jnp.save(f"{directory}/wp_{args.method}_errors_fwd", jnp.asarray(errors_fwd))
jnp.save(f"{directory}/wp_{args.method}_errors_rev", jnp.asarray(errors_rev))
