# todo: use a different differential equation?
# todo: 2d
# todo: have a comparison

import functools

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import optax
from matfree_extensions import arnoldi

# Set discretisation parameters
dx = 1e-1
xs_1 = jnp.arange(0.0, 1.0 + dx, step=dx)
xs_2 = jnp.arange(0.0, 1.0 + dx, step=dx)
xs = jnp.stack(jnp.meshgrid(xs_1, xs_2)).reshape((2, -1))
xs_flat, unflatten_xs = jax.flatten_util.ravel_pytree(xs)


coeff_true = {"init": jnp.asarray([0.5, 0.5]), "rhs": 0.1}
_coeff, unflatten_p = jax.flatten_util.ravel_pytree(coeff_true)

krylov_depth = 10


# Set problem parameters
def init(x, s):
    x = unflatten_xs(x)
    diff = x - s[:, None]
    fx = jax.vmap(lambda d: jnp.exp(-20 * jnp.dot(d, d)), in_axes=-1, out_axes=-1)(diff)
    return jax.flatten_util.ravel_pytree(fx)[0]


# Discretise
stencil = jnp.asarray([[0.0, -1.0, 0.0], [-1, 2.0, -1], [0.0, -1.0, 0.0]]) / dx**2


_, unflatten_y = jax.flatten_util.ravel_pytree(jnp.meshgrid(xs_1, xs_2)[0])


def rhs(x, d):
    x = unflatten_y(x)
    x_padded = jnp.pad(x, 1, mode="constant", constant_values=0.0)

    fx = jax.scipy.signal.convolve2d(-d * stencil, x_padded, mode="valid")
    return jax.flatten_util.ravel_pytree(fx)[0]


algorithm = arnoldi.arnoldi(rhs, krylov_depth, reortho="full", custom_vjp=True)


# Parameter-to-solution/error operators


@jax.jit
@jax.value_and_grad
def parameter_to_error(params, targets):
    solution = parameter_to_solution(1.0, params)
    diff = solution - targets
    return jnp.dot(diff, diff)


@jax.jit
def parameter_to_solution(t, params):
    coeff_ = unflatten_p(params)
    y0 = init(xs, coeff_["init"])
    A = jax.jacfwd(lambda s: rhs(s, coeff_["rhs"]))(y0)
    return jax.scipy.linalg.expm(t * A) @ y0


@jax.jit
def parameter_to_solution_matfree(t, params):
    coeff_ = unflatten_p(params)
    y0 = init(xs, coeff_["init"])
    Q, H, _r, c = algorithm(y0, coeff_["rhs"])
    e1 = jnp.eye(len(H))[0, :]
    return c * Q @ jax.scipy.linalg.expm(t * H) @ e1


# Create an optimization problem
coeff_true_flat, _unflatten = jax.flatten_util.ravel_pytree(coeff_true)
solution_true = parameter_to_solution(1.0, coeff_true_flat)
noise = 1e-5 * jax.random.normal(jax.random.PRNGKey(2), shape=solution_true.shape)
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

    if count % 1 == 0:
        print(count, coeff, jnp.linalg.norm(gradient))
    count += 1


dt = 0.1
ts = jnp.arange(0.0, 1.0 + dt, step=dt)

# Simulation
sol = jax.vmap(lambda s: parameter_to_solution(s, coeff), out_axes=1)(ts)
plt.plot(xs, sol, color="gray", alpha=0.5, label="truth")

# Truth
sol = jax.vmap(lambda s: parameter_to_solution(s, coeff_true_flat), out_axes=1)(ts)
plt.plot(xs, sol, color="blue", alpha=0.5, label="truth")

# Data
plt.plot(xs, data, label="data")


# https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.show()
print(f"Found coefficient coeff={unflatten_p(coeff)}")
