# todo: use a proper optimizer
# todo: matrix-free implementation
# todo: use more parameters
# todo: 2d
# todo: have a comparison

import functools

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import optax

# Set discretisation parameters
dx = 0.01
xs = jnp.arange(0.0, 1.0 + dx, step=dx)


# Set problem parameters
def init(x, s):
    return jnp.exp(-100 * (x - s) ** 2)


coeff_true = {"init": 0.5, "rhs": 0.1}
_coeff, unflatten = jax.flatten_util.ravel_pytree(coeff_true)

# Discretise
stencil = jnp.asarray([-1, 2.0, -1]) / dx**2


def rhs(x, d):
    x_padded = jnp.pad(x, 1, mode="constant", constant_values=0.0)
    return jnp.convolve(-d * stencil, x_padded, mode="valid")


# Parameter-to-solution/error operators


@jax.jit
@jax.value_and_grad
def parameter_to_error(params, targets):
    solution = parameter_to_solution(1.0, params)
    diff = solution - targets
    return jnp.dot(diff, diff) + solution[0] ** 2 + solution[-1] ** 2


@jax.jit
def parameter_to_solution(t, params):
    coeff_ = unflatten(params)
    y0 = init(xs, coeff_["init"])
    A = jax.jacfwd(lambda s: rhs(s, coeff_["rhs"]))(y0)
    return jax.scipy.linalg.expm(t * A) @ y0


# Create an optimization problem
coeff_true_flat, _unflatten = jax.flatten_util.ravel_pytree(coeff_true)
solution_true = parameter_to_solution(1.0, coeff_true_flat)
noise = 1e-3 * jax.random.normal(jax.random.PRNGKey(2), shape=xs.shape)
data = solution_true + noise
loss_value_and_grad = functools.partial(parameter_to_error, targets=data)

# initial guess
noise = 1e-1 * jax.random.normal(jax.random.PRNGKey(2), shape=coeff_true_flat.shape)
coeff = coeff_true_flat + noise

# Optimizer
learning_rate = 1e-2
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(coeff)


# Optimize

value, gradient = loss_value_and_grad(coeff)  # JIT-compile
gradient = jnp.ones_like(gradient)
count = 0
while jnp.linalg.norm(gradient) > jnp.power(jnp.finfo(value.dtype).eps, 0.5):
    value, gradient = loss_value_and_grad(coeff)

    updates, opt_state = optimizer.update(gradient, opt_state)
    coeff = optax.apply_updates(coeff, updates)

    if count % 10 == 0:
        print(count, coeff, jnp.linalg.norm(gradient))
    count += 1


dt = 0.1
ts = jnp.arange(0.0, 1.0 + dt, step=dt)

# Simulation
sol = jax.vmap(lambda s: parameter_to_solution(s, coeff), out_axes=1)(ts)
plt.plot(xs, sol, color="gray")

# Truth
sol = jax.vmap(lambda s: parameter_to_solution(s, coeff_true_flat), out_axes=1)(ts)
plt.plot(xs, sol, color="blue")

# Data
plt.plot(xs, data)
plt.show()
print(f"Found coefficient coeff={unflatten(coeff)}")
