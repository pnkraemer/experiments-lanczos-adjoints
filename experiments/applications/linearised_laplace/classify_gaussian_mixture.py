"""Classify a Gaussian mixture model and calibrated different GGNs."""

# todo:
#  - No more dense GGN matrices
#  - Make fancy table of results
#  - MNIST
#  - ImageNet with a transformer
#  - Run the competition code (laplace-redux?)
#  - Split the script into different smaller scripts once big
#  - Look at diag and nondiag results next to each other in a single plot

import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matfree_extensions import bnn_util, exp_util

# Make directories
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)

# Parse arguments
# todo: add seed-argument to argparse and
#  average results over seeds in dataframe script?
parser = argparse.ArgumentParser()
parser.add_argument("--ggn", type=str)
parser.add_argument("--numerics", type=str)
args = parser.parse_args()
assert args.numerics in ["Lanczos", "Cholesky"]
assert args.ggn in ["full", "diagonal"]

# A bunch of hyperparameters
seed = 1
num_data_in = 100
num_data_out = 100  # OOD
train_num_epochs = 100
train_batch_size = num_data_in
train_lrate = 1e-2
train_print_frequency = 10
calibrate_num_epochs = 100
calibrate_batch_size = num_data_in
calibrate_lrate = 1e-1
calibrate_print_frequency = 10
calibrate_log_alpha_min = 1e-3
numerics_lanczos_rank = 10
numerics_slq_num_samples = 100
numerics_slq_num_batches = 1
evaluate_num_samples = 100
plot_num_linspace = 250
plot_xmin, plot_xmax = -7, 7
plot_figsize = (8, 3)

# Create data
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

# Create model
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


# Calibrate the linearised Laplace

if args.ggn == "diagonal":
    ggn_type = bnn_util.ggn_diag
elif args.ggn == "full":
    ggn_type = bnn_util.ggn_full
else:
    raise ValueError

ggn_fun = ggn_type(
    model_fun=model_apply,
    loss_single=bnn_util.loss_training_cross_entropy_single,
    param_unflatten=unflatten,
)


def unconstrain(a):
    return calibrate_log_alpha_min + jnp.exp(a)


optimizer = optax.adam(calibrate_lrate)

key, subkey = jax.random.split(key, num=2)
log_alpha = jax.random.normal(subkey, shape=())
optimizer_state = optimizer.init(log_alpha)


if args.numerics == "Lanczos":
    logdet_fun = bnn_util.solver_logdet_slq(
        lanczos_rank=numerics_lanczos_rank,
        slq_num_samples=numerics_slq_num_samples,
        slq_num_batches=numerics_slq_num_batches,
    )
    sample_fun = bnn_util.sampler_lanczos(
        lanczos_rank=numerics_lanczos_rank, ggn_fun=ggn_fun, num=evaluate_num_samples
    )
elif args.numerics == "Cholesky":
    logdet_fun = bnn_util.solver_logdet_dense()
    sample_fun = bnn_util.sampler_cholesky(ggn_fun=ggn_fun, num=evaluate_num_samples)
else:
    raise ValueError

sample_fun = jax.jit(sample_fun)
logdet_fun = jax.jit(logdet_fun)

calib_loss = bnn_util.loss_calibration(
    ggn_fun=ggn_fun, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
)
value_and_grad = jax.jit(jax.value_and_grad(calib_loss, argnums=0))


for epoch in range(calibrate_num_epochs):
    # Subsample data
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(
        subkey, a=num_data_in, shape=(calibrate_batch_size,), replace=False
    )

    # Optimisation step
    if args.numerics == "Lanczos":
        key, subkey = jax.random.split(key)
        loss, grad = value_and_grad(
            log_alpha, variables, x_train[idx], y_train[idx], subkey
        )
    elif args.numerics == "Cholesky":
        loss, grad = value_and_grad(log_alpha, variables, x_train[idx], y_train[idx])
    else:
        raise ValueError
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    if epoch % calibrate_print_frequency == 0:
        print(f"Epoch {epoch}, loss {loss:.3f}, alpha {log_alpha:.3f}")
print()


# Linearise the model around the calibrated alpha


def model_linear(sample, v, x):
    """Evaluate the model after linearising around the optimal parameters."""
    fx = model_apply(unflatten(v), x)
    _, jvp = jax.jvp(lambda p: model_apply(unflatten(p), x), (v,), (sample - v,))
    return fx + jvp


# Sample from the posterior
key, subkey = jax.random.split(key)
samples_params = sample_fun(subkey, unconstrain(log_alpha), variables, x_train, y_train)


# Predict (in-distribution)
samples_logits = jax.vmap(lambda s: model_linear(s, variables, x_train))(samples_params)


# Compute metrics (in-distribution)
samples_probs = jax.nn.softmax(samples_logits, axis=-1)
mean_probs = jnp.mean(samples_probs, axis=0)
accuracy = bnn_util.metric_accuracy(probs=mean_probs, labels_hot=y_train)
nll_fun = jax.vmap(lambda s: bnn_util.metric_nll(logits=s, labels_hot=y_train))
samples_nll = nll_fun(samples_logits)
nll = jnp.mean(samples_nll, axis=0)
ece, mce = bnn_util.metric_ece(probs=mean_probs, labels_hot=y_train, num_bins=10)
conf_in = bnn_util.metric_confidence(probs=mean_probs)


# Create out-of-distribution data
key, subkey = jax.random.split(key, num=2)
key_ood1, key_ood2 = jax.random.split(subkey, num=2)
x_ood = 6 * jax.random.rademacher(key_ood1, (num_data_out, 2))
x_ood += 0.5 * jax.random.normal(key_ood2, (num_data_out, 2))

# Predict and compute metrics (out-of-distribution)
samples_logits = jax.vmap(lambda s: model_linear(s, variables, x_ood))(samples_params)
samples_probs = jax.nn.softmax(samples_logits, axis=-1)
mean_probs = jnp.mean(samples_probs, axis=0)
conf_out = bnn_util.metric_confidence(probs=mean_probs)

# Create dictionary with all results (to be saved to a file)
results = {
    r"Accuracy $\uparrow$": accuracy,
    r"NLL $\downarrow$": nll,
    r"ECE $\downarrow$": ece,
    r"MCE $\downarrow$": mce,
    r"Conf. (in-dist) $\uparrow$": conf_in,
    r"Conf. (out-dist) $\downarrow$": conf_out,
}
print(results)

# Save dictionary to file
file_path = f"{directory_results}/results_{args.ggn}_{args.numerics}.pkl"
with open(file_path, "wb") as f:
    pickle.dump(results, f)


# Create plotting grid
x_1d = jnp.linspace(plot_xmin, plot_xmax, num=plot_num_linspace)
x_plot_x, x_plot_y = jnp.meshgrid(x_1d, x_1d)
x_plot = jnp.stack((x_plot_x, x_plot_y)).reshape((2, -1)).T

# Compute marginal standard deviations for plotting inputs
predictive_cov = bnn_util.predictive_cov(
    hyperparam_unconstrain=unconstrain,
    model_fun=model_apply,
    param_unflatten=unflatten,
    ggn_fun=ggn_fun,
)
covs = predictive_cov(log_alpha, variables, x_train, y_train, x_plot)
variances = jax.vmap(jnp.trace)(covs)
stdevs = jnp.sqrt(variances)
stdevs_plot = stdevs.T.reshape((plot_num_linspace, plot_num_linspace))


# Compute labels for plotting inputs
logits_plot = model_apply(unflatten(variables), x_plot)
labels_plot = jax.nn.log_softmax(logits_plot).argmax(axis=-1)
labels_plot = labels_plot.T.reshape((plot_num_linspace, plot_num_linspace))

# Choose a plotting style
style_data = {
    "in": {
        "color": "black",
        "zorder": 1,
        "linestyle": "None",
        "marker": "o",
        "markeredgecolor": "grey",
        "alpha": 0.75,
    },
    "out": {
        "color": "white",
        "zorder": 1,
        "linestyle": "None",
        "marker": "P",
        "markeredgecolor": "black",
        "alpha": 0.75,
    },
}
style_contour = {
    "uq": {"cmap": "viridis", "zorder": 0},
    "bdry": {"vmin": 0, "vmax": 1, "cmap": "seismic", "zorder": 0, "alpha": 0.5},
}


# Plot the results
layout = [["bdry", "uq"]]
_fig, axes = plt.subplot_mosaic(layout, figsize=plot_figsize, constrained_layout=True)

axes["bdry"].set_title("Decision boundary")
axes["bdry"].contourf(x_plot_x, x_plot_y, labels_plot, 3, **style_contour["bdry"])
axes["bdry"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
axes["bdry"].plot(x_ood[:, 0], x_ood[:, 1], **style_data["out"])

axes["uq"].set_title("Standard deviation of logits")
axes["uq"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
axes["uq"].plot(x_ood[:, 0], x_ood[:, 1], **style_data["out"])
cbar = axes["uq"].contourf(x_plot_x, x_plot_y, stdevs_plot, **style_contour["uq"])
plt.colorbar(cbar)

# Save the plot to a file
plt.savefig(f"{directory_fig}/classify_gaussian_mixture.pdf")
