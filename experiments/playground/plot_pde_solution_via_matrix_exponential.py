# todo: use a different differential equation?
# todo: remove constant parameters from param vector
# todo: implement different matrix exponentials
# todo: use a "proper" loss function
# todo: implement a comparison algorithm
# todo: make this example just minimally shinier and it could end up in the paper!


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

krylov_depth = 100
print(xs_flat.shape)


# Set problem parameters
def init(x, s):
    x = unflatten_xs(x)
    diff = x - s[:, None]
    fx = jax.vmap(lambda d: jnp.exp(-50 * jnp.dot(d, d)), in_axes=-1, out_axes=-1)(diff)
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
    return jnp.sqrt(jnp.dot(diff, diff) / diff.size)


@jax.jit
def parameter_to_solution(t, params):
    coeff_ = unflatten_p(params)
    y0 = init(xs, coeff_["init"])
    Q, H, _r, c = algorithm(y0, coeff_["rhs"])
    e1 = jnp.eye(len(H))[0, :]

    H = (H + H.T) / 2
    eigvals, eigvecs = jnp.linalg.eigh(H)

    expm = eigvecs @ jnp.diag(jnp.exp(t * eigvals)) @ eigvecs.T

    # print(expm)
    # print(eigvecs @ jnp.diag(eigvals) @ eigvecs.T)
    # print(H)
    # assert False
    # expm = jax.scipy.linalg.expm(t * H)
    return c * Q @ expm @ e1


# Create an optimization problem
coeff_true_flat, _unflatten = jax.flatten_util.ravel_pytree(coeff_true)
solution_true = parameter_to_solution(1.0, coeff_true_flat)
noise = 1e-8 * jax.random.normal(jax.random.PRNGKey(2), shape=solution_true.shape)
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


def magnitude(x):
    return jnp.log10(jnp.abs(x) + jnp.finfo(x.dtype).eps)


y1 = jax.vmap(lambda t_: parameter_to_solution(t_, coeff_true_flat))(ts)
plot_kwargs = {"vmin": jnp.amin(magnitude(y1)), "vmax": jnp.amax(magnitude(y1))}
print(plot_kwargs)
for t_, y1_, ax in zip(ts, y1, axes):
    ax.set_title(f"$t={t_:2F}$")
    ax.contourf(mesh[0], mesh[1], magnitude(unflatten_y(y1_)), **plot_kwargs)

axes[0].set_ylabel("Truth")


# initial guess
coeff = jnp.asarray([0.8, 0.8, 1e-3])

y1 = jax.vmap(lambda t_: parameter_to_solution(t_, coeff))(ts)
for y1_, ax in zip(y1, axes_before):
    ax.contourf(mesh[0], mesh[1], magnitude(unflatten_y(y1_)), **plot_kwargs)

axes_before[0].set_ylabel("Initial guess")

# Optimizer
learning_rate = 1e-1
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(coeff)


# Optimize


value, gradient = loss_value_and_grad(coeff)  # JIT-compile


gradient = jnp.ones_like(gradient)
count = 0
for count in range(20):
    value, gradient = loss_value_and_grad(coeff)

    updates, opt_state = optimizer.update(gradient, opt_state)
    coeff = optax.apply_updates(coeff, updates)

    print(count, coeff, magnitude(jnp.linalg.norm(gradient)))


y1 = jax.vmap(lambda t_: parameter_to_solution(t_, coeff))(ts)

for y1_, ax in zip(y1, axes_after):
    ax.contourf(mesh[0], mesh[1], magnitude(unflatten_y(y1_)), **plot_kwargs)
axes_after[0].set_ylabel("Recovered")


plt.show()
