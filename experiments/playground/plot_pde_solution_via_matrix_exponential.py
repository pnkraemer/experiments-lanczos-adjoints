# todo: use a proper optimizer
# todo: matrix-free implementation
# todo: use more parameters
# todo: 2d
# todo: have a comparison

import functools

import jax
import jax.numpy as jnp
import jax.scipy.linalg

# Set discretisation parameters
dx = 0.01
dt = 0.1
ts = jnp.arange(0.0, 1.0 + dt, step=dt)
xs = jnp.arange(0.0, 1.0 + dx, step=dx)

# Set problem parameters
y0 = jnp.exp(-100 * (xs - 0.5) ** 2)
coeff_true = 0.1

# Discretise
stencil = jnp.asarray([-1, 2.0, -1]) / dx**2


def matvec(x):
    x_padded = jnp.pad(x, 1, mode="constant", constant_values=0.0)
    return jnp.convolve(stencil, x_padded, mode="valid")


# Parameter-to-solution/error operators


@jax.jit
@jax.value_and_grad
def parameter_to_error(inputs, targets):
    solution = parameter_to_solution(1.0, inputs)
    diff = solution - targets
    return jnp.dot(diff, diff)


@jax.jit
def parameter_to_solution(t, c):
    A = jax.jacfwd(lambda s: matvec(s))(y0)
    return jax.scipy.linalg.expm(-c * t * A) @ y0


# Create an optimization problem

solution_true = parameter_to_solution(1.0, coeff_true)
noise = 0.01 * jax.random.normal(jax.random.PRNGKey(1), shape=y0.shape)
data = solution_true + noise
loss_value_and_grad = functools.partial(parameter_to_error, targets=data)

# Optimize

learning_rate = 0.01
coeff = 0.2  # initial guess
val, grad = loss_value_and_grad(coeff)  # JIT-compile

while jnp.linalg.norm(grad) > jnp.sqrt(jnp.finfo(val.dtype).eps):
    value, gradient = loss_value_and_grad(coeff)

    # Gradient descent
    coeff = coeff - learning_rate * gradient
    print(coeff)
