# todo:
#  - No more dense GGN matrices
#  - Make fancy table of results
#  - MNIST
#  - ImageNet with a transformer
#  - Run the competition code (laplace-redux?)
#  - Split the script into different smaller scripts once big
#  - Look at diag and nondiag results next to each other in a single plot

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matfree_extensions import bnn_util, exp_util

# Make directories
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

# Create data
num_data = 100
key_1, key_2 = jax.random.split(jax.random.PRNGKey(1))
m = 1.15
mu_1 = jnp.array((-m, m))
mu_2 = jnp.array((m, -m))
x_1 = 0.6 * jax.random.normal(key_1, (num_data, 2)) + mu_1[None, :]
y_1 = jnp.asarray(num_data * [[1, 0]])
x_2 = 0.6 * jax.random.normal(key_2, (num_data, 2)) + mu_2[None, :]
y_2 = jnp.asarray(num_data * [[0, 1]])
x_train = jnp.concatenate([x_1, x_2], axis=0)
y_train = jnp.concatenate([y_1, y_2], axis=0)

# Create model
model_init, model_apply = bnn_util.model_mlp(out_dims=2, activation=jnp.tanh)
variables_dict = model_init(jax.random.PRNGKey(42), x_train)
variables, unflatten = jax.flatten_util.ravel_pytree(variables_dict)

# Optimise

optimizer = optax.adam(1e-2)
optimizer_state = optimizer.init(variables)
n_epochs = 100
key = jax.random.PRNGKey(412576)
batch_size = num_data


def loss_p(v, x, y):
    y_pred = model_apply(unflatten(v), x)
    return bnn_util.loss_training_cross_entropy(y_pred, y)


loss_value_and_grad = jax.jit(jax.value_and_grad(loss_p, argnums=0))

for epoch in range(100):
    # Subsample data
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, x_train.shape[0], (batch_size,), replace=False)

    # Optimiser step
    loss, grad = loss_value_and_grad(variables, x_train[idx], y_train[idx])
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    variables = optax.apply_updates(variables, updates)

    # Look at intermediate results
    if epoch % 10 == 0:
        y_pred = model_apply(unflatten(variables), x_train[idx])
        y_probs = jax.nn.softmax(y_pred, axis=-1)
        acc = bnn_util.metric_accuracy(probs=y_probs, labels_hot=y_train[idx])
        print(f"Epoch {epoch}, loss {loss:.3f}, accuracy {acc:.3f}")

print()
# Optimize the calibration loss


def unconstrain(a):
    return 1e-3 + jnp.exp(a)


log_alpha = 0.0
optimizer = optax.adam(1e-1)
optimizer_state = optimizer.init(log_alpha)

ggn_fun = bnn_util.ggn(
    model_fun=model_apply,
    loss_single=bnn_util.loss_training_cross_entropy_single,
    param_unflatten=unflatten,
)

# rank ~10 because GGN rank is 10 in current parametrisation
# logdet_fun = bnn_util.solver_logdet_slq(
#     lanczos_rank=10, slq_num_samples=100, slq_num_batches=1
# )
logdet_fun = bnn_util.solver_logdet_dense()
calib_loss = bnn_util.loss_calibration(
    ggn_fun=ggn_fun, hyperparam_unconstrain=unconstrain, logdet_fun=logdet_fun
)
value_and_grad = jax.jit(jax.value_and_grad(calib_loss, argnums=0))


for epoch in range(100):
    # Subsample data
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(subkey, x_train.shape[0], (batch_size,), replace=False)

    # Optimisation step
    key, subkey = jax.random.split(key)
    loss, grad = value_and_grad(log_alpha, variables, x_train[idx], y_train[idx])
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss {loss:.3f}, alpha {log_alpha:.3f}")
print()

num_linspace = 250
x_1d = jnp.linspace(-4, 4, num=num_linspace)
xs, ys = jnp.meshgrid(x_1d, x_1d)
X = jnp.stack((xs, ys)).reshape((2, -1)).T  # (2500, 2)
predvar = bnn_util.predictive_variance(
    hyperparam_unconstrain=unconstrain,
    model_fun=model_apply,
    param_unflatten=unflatten,
    ggn_fun=ggn_fun,
)


covs = predvar(log_alpha, variables, x_train, y_train, X)

# Sample from the posterior
num_samples = 100
key, subkey = jax.random.split(key)

sampler = bnn_util.sampler_cholesky(ggn_fun, num=num_samples)
samples = sampler(key, unconstrain(log_alpha), variables, x_train, y_train)


# Compute ID metrics
def lin_pred(sample, v, x):
    fx = model_apply(unflatten(v), x)
    _, jvp = jax.jvp(lambda p: model_apply(unflatten(p), x), (v,), (sample - v,))
    return fx + jvp


samples_lin_pred = jax.vmap(lambda s: lin_pred(s, variables, x_train))(samples)


mean_predictive = jnp.mean(samples_lin_pred, axis=0)
probs = jax.nn.softmax(mean_predictive, axis=-1)
accuracy = bnn_util.metric_accuracy(probs=probs, labels_hot=y_train)
print()
print("Accuracy mean before softmax:", accuracy)

probs = jax.nn.softmax(samples_lin_pred, axis=-1)
mean_probs = jnp.mean(probs, axis=0)
accuracy = bnn_util.metric_accuracy(probs=mean_probs, labels_hot=y_train)
print("Accuracy mean after softmax:", accuracy)
print()


# Compute NLL
nll_fun = jax.vmap(lambda s: bnn_util.metric_nll(logits=s, labels_hot=y_train))
nll = nll_fun(samples_lin_pred).mean(axis=0)
print("NLL:", nll)

ece, mce = bnn_util.metric_ece(probs=mean_probs, labels_hot=y_train, num_bins=100)
print("ECE:", ece)


# Plot
figsize = (10, 3)
fig, axes = plt.subplot_mosaic(
    [["uncertainty", "boundary"]], figsize=figsize, constrained_layout=True
)

axes["boundary"].set_title("Decision boundary")
y_pred = model_apply(unflatten(variables), X)
which_class = jax.nn.log_softmax(y_pred).argmax(axis=-1)
values = which_class.T.reshape((num_linspace, num_linspace))
axes["boundary"].contourf(
    xs, ys, values, 3, vmin=0, vmax=1, cmap="seismic", alpha=0.5, zorder=0
)
axes["boundary"].scatter(x_train[:, 0], x_train[:, 1], color="black", zorder=1)

axes["uncertainty"].set_title("Standard deviation of logits")
# variances = jax.vmap(lambda a, b: b[a, a])(which_class, covs)
variances = jax.vmap(jnp.trace)(covs)
z = jnp.sqrt(variances).T.reshape((num_linspace, num_linspace))
axes["uncertainty"].scatter(x_train[:, 0], x_train[:, 1], color="black", zorder=1)
colorbar_values = axes["uncertainty"].contourf(
    xs, ys, z, 50, alpha=0.95, zorder=0, vmin=0, vmax=10
)
plt.colorbar(colorbar_values)
plt.savefig(f"{directory}/classify_gaussian_mixture.pdf")
