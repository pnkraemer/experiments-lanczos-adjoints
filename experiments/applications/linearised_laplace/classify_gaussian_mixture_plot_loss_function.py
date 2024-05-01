"""Classify a Gaussian mixture model and calibrated different GGNs."""

import functools
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matfree_extensions.util import bnn_util, exp_util
from tueplots import axes, figsizes, fontsizes

plt.rcParams.update(fontsizes.neurips2024(default_smaller=2))
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.legend())
plt.rcParams.update(axes.spines(left=True, right=False, bottom=True, top=False))
plt.rcParams.update(
    figsizes.neurips2024(nrows=1, ncols=1, rel_width=0.4, height_to_width_ratio=0.8)
)

# Make directories
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)


def loss_slq(*, num_matvecs, num_samples):
    ggn_function = bnn_util.ggn_full(
        model_fun=model_apply,
        loss_single=bnn_util.loss_training_cross_entropy_single,
        param_unflatten=unflatten,
    )
    logdet_fun = bnn_util.solver_logdet_slq(
        lanczos_rank=num_matvecs, slq_num_samples=num_samples, slq_num_batches=1
    )
    loss_ = loss_function(
        ggn_fun=ggn_function, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
    )

    @jax.jit
    def lossfun(key_, a):
        return loss_(a, variables, x_train, y_train, key_)

    return lossfun


def loss_diagonal():
    ggn_function = bnn_util.ggn_diag(
        model_fun=model_apply,
        loss_single=bnn_util.loss_training_cross_entropy_single,
        param_unflatten=unflatten,
    )
    logdet_fun = bnn_util.solver_logdet_dense()
    loss_ = loss_function(
        ggn_fun=ggn_function, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
    )

    @jax.jit
    def lossfun(a):
        return loss_(a, variables, x_train, y_train)

    return lossfun


def loss_cholesky():
    ggn_function = bnn_util.ggn_full(
        model_fun=model_apply,
        loss_single=bnn_util.loss_training_cross_entropy_single,
        param_unflatten=unflatten,
    )

    logdet_fun = bnn_util.solver_logdet_dense()
    loss_ = loss_function(
        ggn_fun=ggn_function, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
    )

    @jax.jit
    def lossfun(a):
        return loss_(a, variables, x_train, y_train)

    return lossfun


def plot_function(axis, fun, xs, *, color, linewidth, marker, **linestyle):
    axis.plot(
        xs,
        jax.vmap(fun)(xs),
        marker="None",
        color=color,
        linewidth=linewidth,
        **linestyle,
    )

    markerstyle = {
        "marker": marker,
        "linestyle": "None",
        "color": "white",
        "markeredgecolor": color,
        "markeredgewidth": linewidth / 2,
        "markersize": 4,
    }
    i = jnp.argmin(jax.vmap(fun)(xs))
    axis.plot(xs[i], (jax.vmap(fun)(xs))[i], **markerstyle, zorder=150)


def plot_gradient(axis, x, f, df, *, color):
    xs = jnp.linspace(x - 0.1, x + 0.1)

    def linear(z):
        return f + df * (z - x)

    arrow_begin = (xs[-1], linear(xs[-1]))
    arrow_end = (xs[0], linear(xs[0]))
    props = {"facecolor": color, "arrowstyle": "->", "alpha": 0.9}
    axis.annotate(
        "",
        xy=arrow_begin,
        xytext=arrow_end,
        arrowprops=props,
        annotation_clip=True,
        zorder=100,
    )


# A bunch of hyperparameters
seed = 1
num_data_in = 100
train_num_epochs = 100
train_batch_size = num_data_in
train_lrate = 1e-2
train_print_frequency = 10
calibrate_log_alpha_min = 1e-10
lanczos_rank_bad = 3
lanczos_rank_good = 10
slq_num_samples_bad = 2
slq_num_samples_good = 10
plot_num_samples_lanczos = 3
plot_lw_bg, plot_alpha_bg = 1.2, 0.4
plot_lw_fg, plot_alpha_fg = 1.2, 0.75

# Create the data
key = jax.random.PRNGKey(seed)
key, key_1, key_2 = jax.random.split(key, num=3)
m = 1.15
mu_1, mu_2 = jnp.array((-m, m)), jnp.array((m, -m))
x_1 = 0.6 * jax.random.normal(key_1, (num_data_in, 2)) + mu_1[None, :]
y_1 = jnp.asarray(num_data_in * [[1, 0]])
x_2 = 0.6 * jax.random.normal(key_2, (num_data_in, 2)) + mu_2[None, :]
y_2 = jnp.asarray(num_data_in * [[0, 1]])
x_train = jnp.concatenate([x_1, x_2], axis=0)
y_train = jnp.concatenate([y_1, y_2], axis=0)

# Create the model
model_init, model_apply = bnn_util.model_mlp(out_dims=2, activation=jnp.tanh)
key, subkey = jax.random.split(key, num=2)
variables_dict = model_init(subkey, x_train)
variables, unflatten = jax.flatten_util.ravel_pytree(variables_dict)

print("Number of parameters:", variables.size)
# Train the model

optimizer = optax.adam(train_lrate)
optimizer_state = optimizer.init(variables)


def loss_p(v, x, y):
    logits = model_apply(unflatten(v), x)
    return bnn_util.loss_training_cross_entropy(logits=logits, labels_hot=y)


loss_value_and_grad = jax.jit(jax.value_and_grad(loss_p, argnums=0))

for epoch in range(train_num_epochs):
    # Subsample data
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(
        subkey, x_train.shape[0], (train_batch_size,), replace=False
    )

    # Apply an optimizer-step
    loss, grad = loss_value_and_grad(variables, x_train[idx], y_train[idx])
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    variables = optax.apply_updates(variables, updates)

    # Look at intermediate results
    if epoch % train_print_frequency == 0:
        y_pred = model_apply(unflatten(variables), x_train[idx])
        y_probs = jax.nn.softmax(y_pred, axis=-1)
        acc = bnn_util.metric_accuracy(probs=y_probs, labels_hot=y_train[idx])
        print(f"Epoch {epoch}, loss {loss:.3f}, accuracy {acc:.3f}")

print()


# Set up the linearised Laplace
loss_function = bnn_util.loss_calibration


def unconstrain(a):
    # return calibrate_log_alpha_min + jnp.exp(a)
    return calibrate_log_alpha_min + a**2


alpha0 = 0.3
log_alphas = jnp.linspace(0.1, 1.1)

fig, ax = plt.subplots(dpi=200)
ax.set_xlim((jnp.amin(log_alphas), jnp.amax(log_alphas)))


# axes["bad"].set_ylim((-1, 245))


print("Plotting full GGN + Lanczos (bad approximation)")
key, subkey = jax.random.split(key, num=2)
subkeys = jax.random.split(subkey, num=plot_num_samples_lanczos)
slq_bad_fun = loss_slq(num_matvecs=lanczos_rank_bad, num_samples=slq_num_samples_bad)
slq_bad_label = f"{lanczos_rank_bad} MVs, {slq_num_samples_bad} samples"
slq_bad_style = {"linewidth": plot_lw_fg, "alpha": plot_alpha_fg}
for k in subkeys:
    k0, k1 = jax.random.split(k, num=2)

    fun_bad = functools.partial(slq_bad_fun, k0)
    plot_function(
        ax,
        fun_bad,
        log_alphas,
        color="steelblue",
        label=slq_bad_label,
        marker="X",
        **slq_bad_style,
    )

    x0 = alpha0 + 0.1 * jax.random.normal(k1, shape=())
    f0, df0 = fun_bad(x0), jax.grad(fun_bad)(x0)
    plot_gradient(ax, x0, f0, df0, color="black")


bbox = {"boxstyle": "round, pad=0.05", "facecolor": "white", "edgecolor": "None"}
annotate_kwargs = {
    "bbox": bbox,
    "arrowprops": {"arrowstyle": "->"},
    "color": "black",
    "fontsize": "x-small",
}
ax.annotate("New gradients!", xy=(x0, f0), xytext=(0.15, 170), **annotate_kwargs)

print("Plotting full GGN + Lanczos (good approximation)")
key, subkey = jax.random.split(key, num=2)
subkeys = jax.random.split(subkey, num=plot_num_samples_lanczos)
slq_good_fun = loss_slq(num_matvecs=lanczos_rank_good, num_samples=slq_num_samples_good)
slq_good_label = f"{lanczos_rank_good} MVs, {slq_num_samples_good} samples"
slq_good_style = {"linewidth": plot_lw_fg, "alpha": plot_alpha_fg}
for k in subkeys:
    k0, k1 = jax.random.split(k, num=2)

    fun_good = functools.partial(slq_good_fun, k0)
    plot_function(
        ax,
        fun_good,
        log_alphas,
        color="firebrick",
        label=slq_good_label,
        marker="^",
        **slq_good_style,
    )

    x0 = alpha0 + 0.1 * jax.random.normal(k1, shape=())
    f0, df0 = fun_good(x0), jax.grad(fun_good)(x0)
    plot_gradient(ax, x0, f0, df0, color="black")


print("Plotting diagonal approximation")
diagonal_fun = loss_diagonal()
diagonal_style = {
    "color": "gray",
    "linestyle": "dotted",
    "linewidth": plot_lw_bg,
    "alpha": plot_alpha_bg,
    "marker": "P",
}
plot_function(ax, diagonal_fun, log_alphas, **diagonal_style)
ax.annotate(
    "Diagonal approximation",
    (0.5, diagonal_fun(0.5)),
    bbox=bbox,
    color="gray",
    fontsize="x-small",
)

print("Plotting the truth")
true_fun = loss_cholesky()

print("Plotting the vertical line for the true optimum")
inputs, targets = log_alphas, jax.vmap(true_fun)(log_alphas)
idx = jnp.argmin(targets, axis=0)
ax.axvline(inputs[idx], linewidth=0.5, color="black")
ax.annotate("Optimum", xy=(inputs[idx] + 0.01, 130), fontsize="x-small", color="black")


print("Setting the legend")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize="xx-small")


ax.set_xlabel("Hyperparameter", fontsize="medium")
ax.set_ylabel("Calibration loss", fontsize="medium")


# Save the plot to a file
plt.savefig(f"{directory_fig}/slq_versus_diagonal_loss.pdf")

plt.show()
