# todo: use a different differential equation?
# todo: remove constant parameters from param vector
# todo: implement different matrix exponentials
# todo: use a "proper" loss function
# todo: implement a comparison algorithm
# todo: make this example just minimally shinier and it could end up in the paper!


import os

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
import matplotlib.pyplot as plt
import optax
from matfree_extensions.util import exp_util, pde_util
from tueplots import axes, figsizes, fontsizes

# Set a few display-related parameters
plt.rcParams.update(figsizes.icml2024_full(ncols=5, nrows=3, height_to_width_ratio=1.0))
plt.rcParams.update(fontsizes.icml2024())
plt.rcParams.update(axes.lines())


def magnitude(x):
    return jnp.log10(jnp.abs(x) + jnp.finfo(x.dtype).eps)


if __name__ == "__main__":
    # Set parameters
    dx = 2.5e-2
    krylov_depth = 10
    seed = 1
    num_epochs = 25

    # Random key
    key = jax.random.PRNGKey(seed)

    # Set discretisation parameters
    xs_1d = jnp.arange(0.0, 1.0 + dx, step=dx)
    mesh = pde_util.mesh_2d_tensorproduct(xs_1d, xs_1d)
    print(f"Number of points: {mesh.size // 2}")
    # Initial condition
    pde_init, _p_init = pde_util.pde_2d_init_bell()

    # Right-hand side
    stencil = pde_util.stencil_2d_laplacian(dx)
    pde_rhs, _p_rhs = pde_util.pde_2d_rhs_laplacian(stencil=stencil)

    # Solution operator
    expm_fun = pde_util.expm_arnoldi(krylov_depth, reortho="full", custom_vjp=True)
    solution_operator = pde_util.solution_terminal(
        init=pde_init, rhs=pde_rhs, expm=expm_fun
    )
    loss_fun = pde_util.loss_mse()

    # Data
    p_init = {"center": jnp.asarray([0.5, 0.5])}
    p_rhs = {"intensity_sqrt": 0.125}
    param_to_solution_true = solution_operator(p_init=p_init, p_rhs=p_rhs)
    data = param_to_solution_true(1.0, mesh)

    # PyTree structure of parameters
    p_like = (p_init, p_rhs)
    p_flat, unflatten_p = jax.flatten_util.ravel_pytree(p_like)

    @jax.value_and_grad
    def loss_value_and_grad(p):
        evaluated = solution_operator(*unflatten_p(p))(1.0, mesh)
        return loss_fun(evaluated, targets=data)

    # Initial guess
    params_before = jnp.asarray([0.8, 0.8, 1e-3])
    params_after = params_before

    # Optimizer
    learning_rate = 1e-1
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_after)

    # Optimize

    value, gradient = loss_value_and_grad(params_after)  # JIT-compile
    gradient = jnp.ones_like(gradient)
    count = 0
    for count in range(num_epochs):
        value, gradient = loss_value_and_grad(params_after)

        updates, opt_state = optimizer.update(gradient, opt_state)
        params_after = optax.apply_updates(params_after, updates)
        print(count, unflatten_p(params_after))

    # Plot

    ts = jnp.arange(0.0, 1.2, step=0.2)

    fig, (axes, axes_before, axes_after) = plt.subplots(
        nrows=3, ncols=len(ts), sharex=True, sharey=True
    )
    fig.suptitle(f"N={mesh.size //2} points; K={krylov_depth} Krylov-depth")

    axes[0].set_ylabel("Truth")
    y1 = jax.vmap(param_to_solution_true, in_axes=(0, None))(ts, mesh)
    plot_kwargs = {"vmin": jnp.amin(magnitude(y1)), "vmax": jnp.amax(magnitude(y1))}
    for t_, y1_, ax in zip(ts, y1, axes):
        ax.set_title(f"$t={t_:2F}$")
        ax.contourf(mesh[0], mesh[1], magnitude(y1_), **plot_kwargs)

    axes_before[0].set_ylabel("Before optimisation")
    param_to_solution_approx = solution_operator(*unflatten_p(params_before))
    y1 = jax.vmap(param_to_solution_approx, in_axes=(0, None))(ts, mesh)
    for y1_, ax in zip(y1, axes_before):
        ax.contourf(mesh[0], mesh[1], magnitude(y1_), **plot_kwargs)

    axes_after[0].set_ylabel("After optimisation")
    param_to_solution_approx = solution_operator(*unflatten_p(params_after))
    y1 = jax.vmap(param_to_solution_approx, in_axes=(0, None))(ts, mesh)
    for y1_, ax in zip(y1, axes_after):
        ax.contourf(mesh[0], mesh[1], magnitude(y1_), **plot_kwargs)

    # Save figure
    directory = exp_util.matching_directory(__file__, "figures/")
    os.makedirs(directory, exist_ok=True)

    plt.savefig(f"{directory}/figure.pdf")
