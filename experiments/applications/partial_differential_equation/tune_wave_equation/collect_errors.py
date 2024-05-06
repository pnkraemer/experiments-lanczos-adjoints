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
args = parser.parse_args()
print(args)

# Load data
directory = exp_util.matching_directory(__file__, "data/")
os.makedirs(directory, exist_ok=True)
path = f"{directory}{args.resolution}x{args.resolution}"

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
val, grad = (loss(parameter, inputs, targets))
val.block_until_ready()
grad.block_until_ready()

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

num_matvecs = 10
expm = pde_util.expm_arnoldi(num_matvecs)
solve = pde_util.solver_expm(0., 1., vector_field, expm=expm)

loss = loss_function(solve)

# Precompile
val, grad = (loss(parameter, inputs, targets))
val.block_until_ready()
grad.block_until_ready()

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
