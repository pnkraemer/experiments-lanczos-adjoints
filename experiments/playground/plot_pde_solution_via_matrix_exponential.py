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
dx = 1e-2
xs_1 = jnp.arange(0.0, 1.0 + dx, step=dx)
xs_2 = jnp.arange(0.0, 1.0 + dx, step=dx)
mesh = jnp.stack(jnp.meshgrid(xs_1, xs_2))
xs = mesh.reshape((2, -1))
xs_flat, unflatten_xs = jax.flatten_util.ravel_pytree(xs)


coeff_true = {"init": jnp.asarray([0.5, 0.5]), "rhs": 0.01}
_coeff, unflatten_p = jax.flatten_util.ravel_pytree(coeff_true)

krylov_depth = 50


# Set problem parameters
def init(x, s):
    x = unflatten_xs(x)
    diff = x - s[:, None]
    fx = jax.vmap(lambda d: jnp.exp(-100 * jnp.dot(d, d)), in_axes=-1, out_axes=-1)(
        diff
    )
    return jax.flatten_util.ravel_pytree(fx)[0]


# Discretise
stencil = jnp.asarray([[0.0, -1.0, 0.0], [-1, 2.0, -1], [0.0, -1.0, 0.0]]) / dx**2


_, unflatten_y = jax.flatten_util.ravel_pytree(jnp.meshgrid(xs_1, xs_2)[0])


@jax.jit
def rhs(x, d):
    d = 1e-3
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


#
# @jax.jit
# def parameter_to_solution(t, params):
#     coeff_ = unflatten_p(params)
#     y0 = init(xs, coeff_["init"])
#     A = jax.jacfwd(lambda s: rhs(s, coeff_["rhs"]))(y0)
#     return jax.scipy.linalg.expm(t * A) @ y0
#


@jax.jit
def parameter_to_solution(t, params):
    coeff_ = unflatten_p(params)
    y0 = init(xs, coeff_["init"])
    Q, H, _r, c = algorithm(y0, coeff_["rhs"])
    e1 = jnp.eye(len(H))[0, :]
    return c * Q @ jax.scipy.linalg.expm(t * H) @ e1


# Create an optimization problem
coeff_true_flat, _unflatten = jax.flatten_util.ravel_pytree(coeff_true)
solution_true = parameter_to_solution(1.0, coeff_true_flat)
noise = 1e-3 * jax.random.normal(jax.random.PRNGKey(2), shape=solution_true.shape)
data = solution_true + noise
loss_value_and_grad = functools.partial(parameter_to_error, targets=data)


print("....", loss_value_and_grad(coeff_true_flat))


y0 = init(xs_flat, coeff_true["init"])


ts = jnp.arange(0.0, 1.2, step=0.2)

fig, (axes, axes_before, axes_after) = plt.subplots(
    nrows=3,
    ncols=len(ts),
    figsize=(len(ts) * 2, 5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
fig.suptitle(f"N={xs.size //2} points; K={krylov_depth} Krylov-depth")


y1 = jax.vmap(lambda t_: parameter_to_solution(t_, coeff_true_flat))(ts)
for y1_, ax in zip(y1, axes):
    ax.contourf(mesh[0], mesh[1], unflatten_y(y1_))

axes[0].set_ylabel("Truth")


# initial guess
noise = 1e-1 * jax.random.normal(jax.random.PRNGKey(2), shape=coeff_true_flat.shape)
coeff = coeff_true_flat + noise
coeff = jnp.asarray([0.7, 0.7, 1e-3])


y1 = jax.vmap(lambda t_: parameter_to_solution(t_, coeff))(ts)
for t_, y1_, ax in zip(ts, y1, axes_before):
    ax.set_title(f"$t={t_}$")
    ax.contourf(mesh[0], mesh[1], unflatten_y(y1_))

axes_before[0].set_ylabel("Truth")

# Optimizer
learning_rate = 1e-2
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(coeff)


# Optimize


value, gradient = loss_value_and_grad(coeff)  # JIT-compile


gradient = jnp.ones_like(gradient)
count = 0
for count in range(10):
    value, gradient = loss_value_and_grad(coeff)

    updates, opt_state = optimizer.update(gradient, opt_state)
    coeff = optax.apply_updates(coeff, updates)

    print(count, coeff, jnp.linalg.norm(gradient))


y1 = jax.vmap(lambda t_: parameter_to_solution(t_, coeff))(ts)

for y1_, ax in zip(y1, axes_after):
    ax.contourf(mesh[0], mesh[1], unflatten_y(y1_))
axes_after[0].set_ylabel("Recovered")

plt.show()
