"""Evaluate work vs. precision of value_and_grad of matrix exponentials."""

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


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_runs", type=int, required=True, help="Time a function.")
parser.add_argument("--resolution", type=int, required=True, help="Eg. 4, 16, 32, ...")
parser.add_argument("--log2_num_matvec_min", type=int, required=True)
parser.add_argument("--method", type=str, required=True, help="Eg. 'arnoldi'")
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
    print(solver)
    nugget = (jnp.finfo(targets).eps)
    loss = pde_util.loss_mse_relative(nugget=nugget)
    k = jax.random.PRNGKey(1421)
    u = jax.random.uniform(k, shape=mesh.shape)
    def fun(p, y0):
        """Compute the norm of the solution."""
        y1 = solver(y0, p)
        return y1
        # return jnp.vdot(u, y1)

    return jax.jit(jax.value_and_grad(fun))



# Create a reference solver
method, adjoint = "dopri8", "direct"
kwargs = {"num_steps": 1024, "method": method, "adjoint": adjoint}
solve = pde_util.solver_diffrax(0., 1., vector_field, **kwargs)
loss = loss_function(solve)

# Precompile
value, gradient = (loss(parameter, inputs[0]))
value.block_until_ready()
gradient.block_until_ready()
print(value)

ts = []
for _ in range(args.num_runs):
    t0 = time.perf_counter()
    val, grad = loss(parameter, inputs[0])
    val.block_until_ready()
    grad.block_until_ready()
    t1 = time.perf_counter()
    ts.append(t1 - t0)

ts = jnp.stack(ts)

num_matvecs = 2**jnp.arange(args.log2_num_matvec_min, 8)
errors_fwd = []
errors_rev = []
ts_all = []

for nmv in tqdm.tqdm(num_matvecs):
    nmv = int(nmv)
    if nmv > parameter.size:
        break 

    if args.method == "arnoldi":
        expm = pde_util.expm_arnoldi(nmv)
        solve = pde_util.solver_expm(0., 1., vector_field, expm=expm)

    elif args.method == "diffrax:euler+backsolve":
        method, adjoint = "euler", "backsolve"
        kwargs = {"num_steps": nmv, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0., 1., vector_field, **kwargs)

    elif args.method == "diffrax:heun+recursive_checkpoint":
        if nmv < 2:
            raise ValueError
        method, adjoint = "heun", "recursive_checkpoint"
        kwargs = {"num_steps": nmv // 2, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0., 1., vector_field, **kwargs)

    elif args.method == "diffrax:dopri5+recursive_checkpoint":
        if nmv < 5:
            raise ValueError
        method, adjoint = "dopri5", "recursive_checkpoint"
        kwargs = {"num_steps": 1, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0., 1., vector_field, **kwargs)

    elif args.method == "diffrax:tsit5+backsolve":
        if nmv < 5:
            raise ValueError
        method, adjoint = "tsit5", "backsolve"
        kwargs = {"num_steps": nmv // 5, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0., 1., vector_field, **kwargs)

    else:
        msg = f"The method {args.method} is not supported."
        raise ValueError(msg)


    # Compute values and gradients (and precompile while we're at it)
    loss = loss_function(solve)
    f, df = (loss(parameter, inputs[0]))
    print(f)

    # Compute the error
    nugget = (jnp.finfo(targets).eps)
    error = pde_util.loss_mse_relative(nugget=nugget)
    fwd = error(f, targets=value)
    rev = error(df, targets=gradient)
    errors_fwd.append(jnp.sqrt(fwd))
    errors_rev.append(jnp.sqrt(rev))

    # Compute the run times
    ts = []
    for _ in range(args.num_runs):
        t0 = time.perf_counter()
        val, grad = loss(parameter, inputs[0])
        val.block_until_ready()
        grad.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)

    ts = jnp.stack(ts)
    ts_all.append(ts)

    # Clear caches in the hopes of avoiding memory issues?
    jax.clear_caches()
    print(errors_fwd)
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)

jnp.save(f"{directory}/wp_{args.method}_Ns", num_matvecs)
jnp.save(f"{directory}/wp_{args.method}_ts", jnp.asarray(ts_all))
jnp.save(f"{directory}/wp_{args.method}_errors_fwd", jnp.asarray(errors_fwd))
jnp.save(f"{directory}/wp_{args.method}_errors_rev", jnp.asarray(errors_rev))
