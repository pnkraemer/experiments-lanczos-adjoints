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
    nugget = jnp.sqrt(jnp.finfo(targets).eps)
    loss = pde_util.loss_mse_relative(nugget=nugget)
    def fun(p, ins, targ):
        outs = jax.vmap(solver, in_axes=(0, None))(ins, p)
        return loss(outs, targets=targ)
    return jax.jit(jax.value_and_grad(fun))



# Create a reference solver
expm = pde_util.expm_pade()
solve = pde_util.solver_expm(0., 1., vector_field, expm=expm)
loss = loss_function(solve)

# Precompile
value, gradient = (loss(parameter, inputs, targets))
value.block_until_ready()
gradient.block_until_ready()

ts = []
for _ in range(args.num_runs):
    t0 = time.perf_counter()
    val, grad = loss(parameter, inputs, targets)
    val.block_until_ready()
    grad.block_until_ready()
    t1 = time.perf_counter()
    ts.append(t1 - t0)

ts = jnp.stack(ts)
print(ts)

num_matvecs = 2**jnp.arange(3, 8)
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
    elif args.method == "diffrax:tsit5+backsolve":
        if nmv < 5:
            raise ValueError
        method, adjoint = "tsit5", "backsolve"
        kwargs = {"num_steps": nmv // 5, "method": method, "adjoint": adjoint}
        solve = pde_util.solver_diffrax(0., 1., vector_field, **kwargs)
    else:
        raise ValueError


    # Compute values and gradients (and precompile while we're at it)
    loss = loss_function(solve)
    f, df = (loss(parameter, inputs, targets))

    # Compute the error
    nugget = jnp.sqrt(jnp.finfo(targets).eps)
    error = pde_util.loss_mse_relative(nugget=nugget)
    fwd = error(f, targets=value)
    rev = error(df, targets=gradient)
    errors_fwd.append(jnp.sqrt(fwd))
    errors_rev.append(jnp.sqrt(rev))

    # Compute the run times
    ts = []
    for _ in range(args.num_runs):
        t0 = time.perf_counter()
        val, grad = loss(parameter, inputs, targets)
        val.block_until_ready()
        grad.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)

    ts = jnp.stack(ts)
    ts_all.append(ts)
    

directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)

jnp.save(f"{directory}/wp_{args.method}_Ns", num_matvecs)
jnp.save(f"{directory}/wp_{args.method}_ts", jnp.asarray(ts_all))
jnp.save(f"{directory}/wp_{args.method}_errors_fwd", jnp.asarray(errors_fwd))
jnp.save(f"{directory}/wp_{args.method}_errors_rev", jnp.asarray(errors_rev))
