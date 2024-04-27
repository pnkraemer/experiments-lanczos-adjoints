"""Classify a Gaussian mixture model and calibrated different GGNs."""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matfree_extensions.util import bnn_util, exp_util
from tueplots import axes, bundles

plt.rcParams.update(bundles.neurips2024(ncols=2, nrows=1, family="sans-serif"))
plt.rcParams.update(axes.lines())

# Make directories
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)

# A bunch of hyperparameters
seed = 1
num_data_in = 100
train_num_epochs = 100
train_batch_size = num_data_in
train_lrate = 1e-2
train_print_frequency = 10
calibrate_log_alpha_min = 1e-5
numerics_lanczos_rank_bad = 3
numerics_lanczos_rank_good = 10
numerics_slq_num_samples_bad = 2
numerics_slq_num_samples_good = 10
numerics_slq_num_batches = 1
plot_num_linspace = 100
plot_num_samples_lanczos = 3

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


def unconstrain(a):
    return calibrate_log_alpha_min + jnp.exp(a)


log_alpha = -3.5
log_alphas = jnp.linspace(-5.0, 1.0, num=plot_num_linspace)
tangent_ins = jnp.linspace(log_alpha - 0.8, log_alpha + 0.8, num=plot_num_linspace)


fig, axes = plt.subplot_mosaic([["bad", "good"]], sharex=True, sharey=True, dpi=150)
axes["bad"].set_xlim((jnp.amin(log_alphas), jnp.amax(log_alphas)))
axes["bad"].set_ylim((-10, 260))


# First: Full GGN + Lanczos (bad approximation)

ggn_fun = bnn_util.ggn_full(
    model_fun=model_apply,
    loss_single=bnn_util.loss_training_cross_entropy_single,
    param_unflatten=unflatten,
)


def calibration_loss(a, k):
    logdet_fun = bnn_util.solver_logdet_slq(
        lanczos_rank=numerics_lanczos_rank_bad,
        slq_num_samples=numerics_slq_num_samples_bad,
        slq_num_batches=numerics_slq_num_batches,
    )

    loss = bnn_util.loss_calibration(
        ggn_fun=ggn_fun, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
    )
    return loss(a, variables, x_train, y_train, k)


loss_vmap = jax.vmap(calibration_loss, in_axes=(0, None))
value_and_grad = jax.value_and_grad(calibration_loss, argnums=0)

loss_vmap = jax.vmap(loss_vmap, in_axes=(None, 0), out_axes=-1)
value_and_grad = jax.vmap(value_and_grad, in_axes=(None, 0), out_axes=-1)

key, subkey = jax.random.split(key, num=2)
subkeys = jax.random.split(subkey, num=plot_num_samples_lanczos)

values = loss_vmap(log_alphas, subkeys)
value, grad = value_and_grad(log_alpha, subkeys)
tangent_outs = value[None, :] + grad[None, :] * (tangent_ins[:, None] - log_alpha)


idx_opt = jnp.argmin(values, axis=0)
alpha_opt = log_alphas[idx_opt]
values_opt = values[idx_opt]
annotate_loc = (-4.75, 1.1 * jnp.amax(loss_vmap(-4.5 * jnp.ones((1, 1)), subkeys)))
axes["bad"].annotate(
    f"{plot_num_samples_lanczos} SLQ seeds",
    annotate_loc,
    bbox={"boxstyle": "round, pad=0.05", "facecolor": "white", "edgecolor": "None"},
    color="crimson",
    fontsize="small",
)
axes["bad"].plot(log_alphas, values, linewidth=1, color="crimson")

axes["bad"].plot(
    tangent_ins,
    tangent_outs[:, 0],
    zorder=10,
    linestyle="dashed",
    linewidth=2,
    color="black",
)
axes["bad"].annotate(
    "Gradient (new)",
    (tangent_ins[plot_num_linspace // 2], 3 + tangent_outs[plot_num_linspace // 2, 0]),
    color="black",
    weight="bold",
    fontsize="small",
    bbox={"boxstyle": "round, pad=0.05", "facecolor": "white", "edgecolor": "None"},
)


# Now the good approximation

ggn_fun = bnn_util.ggn_full(
    model_fun=model_apply,
    loss_single=bnn_util.loss_training_cross_entropy_single,
    param_unflatten=unflatten,
)


def calibration_loss(a, k):
    logdet_fun = bnn_util.solver_logdet_slq(
        lanczos_rank=numerics_lanczos_rank_good,
        slq_num_samples=numerics_slq_num_samples_good,
        slq_num_batches=numerics_slq_num_batches,
    )

    loss = bnn_util.loss_calibration(
        ggn_fun=ggn_fun, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
    )
    return loss(a, variables, x_train, y_train, k)


loss_vmap = jax.vmap(calibration_loss, in_axes=(0, None))
value_and_grad = jax.value_and_grad(calibration_loss, argnums=0)

loss_vmap = jax.vmap(loss_vmap, in_axes=(None, 0), out_axes=-1)
value_and_grad = jax.vmap(value_and_grad, in_axes=(None, 0), out_axes=-1)

key, subkey = jax.random.split(key, num=2)
subkeys = jax.random.split(subkey, num=plot_num_samples_lanczos)

values = loss_vmap(log_alphas, subkeys)
value, grad = value_and_grad(log_alpha, subkeys)
tangent_outs = value[None, :] + grad[None, :] * (tangent_ins[:, None] - log_alpha)


idx_opt = jnp.argmin(values, axis=0)
alpha_opt = log_alphas[idx_opt]
values_opt = values[idx_opt]
annotate_loc = (
    -4.75,
    10 + 1.05 * jnp.amax(loss_vmap(-4.5 * jnp.ones((1, 1)), subkeys)),
)
axes["good"].annotate(
    f"{plot_num_samples_lanczos} SLQ seeds",
    annotate_loc,
    color="crimson",
    fontsize="small",
)
axes["good"].plot(log_alphas, values, linewidth=1, color="crimson")

axes["good"].plot(
    tangent_ins,
    tangent_outs[:, 0],
    zorder=10,
    linestyle="dashed",
    linewidth=2,
    color="black",
)
# axes["good"].annotate(
#     "Contrib.",
#     (tangent_ins[0] - 0.7, tangent_outs[0, 0] - 20.0),
#     color="black",
#     weight="bold",
# )


# Second: Full GGN + Cholesky


def calibration_loss(a):
    logdet_fun = bnn_util.solver_logdet_dense()
    loss = bnn_util.loss_calibration(
        ggn_fun=ggn_fun, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
    )
    return loss(a, variables, x_train, y_train)


loss_vmap = jax.vmap(calibration_loss)
value_and_grad = jax.value_and_grad(calibration_loss, argnums=0)

values = loss_vmap(log_alphas)
value, grad = value_and_grad(log_alpha)
tangent_outs = value + grad * (tangent_ins - log_alpha)


idx_opt = jnp.argmin(values, axis=0)
alpha_opt = log_alphas[idx_opt]
values_opt = values[idx_opt]

annotate_loc = (-4.8, -30 + jnp.amax(loss_vmap(-4.8 * jnp.ones((1, 1)))))
for ax in [axes["bad"], axes["good"]]:
    ax.annotate("Truth", annotate_loc, color="grey", fontsize="small")
    ax.plot(log_alphas, values, linewidth=3, alpha=0.5, color="grey")
# plt.plot(tangent_ins, tangent_outs, color="C1")

# axes["bad"].set_xticks([-5, -3.5, -2, alpha_opt, -0.5, 1], [-5, -3.5, -2, r"Opt.", -0.5, 1])

# Third: Diagonal GGN

ggn_fun = bnn_util.ggn_diag(
    model_fun=model_apply,
    loss_single=bnn_util.loss_training_cross_entropy_single,
    param_unflatten=unflatten,
)


def calibration_loss(a):
    logdet_fun = bnn_util.solver_logdet_dense()
    loss = bnn_util.loss_calibration(
        ggn_fun=ggn_fun, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
    )
    return loss(a, variables, x_train, y_train)


loss_vmap = jax.vmap(calibration_loss)
value_and_grad = jax.value_and_grad(calibration_loss, argnums=0)

values = loss_vmap(log_alphas)
value, grad = value_and_grad(log_alpha)
tangent_outs = value + grad * (tangent_ins - log_alpha)

idx_opt = jnp.argmin(values, axis=0)
alpha_opt = log_alphas[idx_opt]
values_opt = values[idx_opt]
# plt.plot(
#     alpha_opt,
#     values_opt,
#     marker=".",
#     color="grey",
# )

annotate_loc = (-4, 1.05 * jnp.amax(loss_vmap(-4 * jnp.ones((1, 1)))))
for ax in [axes["bad"], axes["good"]]:
    ax.annotate("Diagonal approximation", annotate_loc, color="black", fontsize="small")
    ax.plot(log_alphas, values, color="black", linestyle="dotted")
# plt.plot(tangent_ins, tangent_outs, color="C2")


# Remove duplicate legend entries
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())


axes["bad"].set_ylabel("Calibration loss")
axes["bad"].set_xlabel("Parameter (log)")
axes["good"].set_xlabel("Parameter (log)")

axes["bad"].set_title(
    f"SLQ: {numerics_lanczos_rank_bad} matvecs, {numerics_slq_num_samples_bad} samples per seed",
    fontsize="medium",
)
axes["good"].set_title(
    rf"SLQ: $\geq$ {numerics_lanczos_rank_good} matvecs, $\geq$ {numerics_slq_num_samples_good} samples per seed",
    fontsize="medium",
)

# Save the plot to a file
plt.savefig(f"{directory_fig}/slq_versus_diagonal_loss.pdf")

plt.show()
