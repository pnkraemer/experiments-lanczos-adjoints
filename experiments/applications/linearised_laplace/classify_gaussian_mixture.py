# todo:
#  - Use them to evaluate the calibration result
#  - Use sparse linear algebra
#  - Plot solutions
#  - Move some code to the src
#  - Split the script into different smaller scripts once big


import os

import jax
import jax.numpy as jnp
import optax
from matfree_extensions import bnn_util, exp_util

# Make directories
directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

# Create data
num_data = 100
key_1, key_2 = jax.random.split(jax.random.PRNGKey(1))
m = 2.0
mu_1 = jnp.array((-m, -m))
mu_2 = jnp.array((m, m))
x_1 = jax.random.normal(key_1, (num_data, 2)) + mu_1[None, :]
y_1 = jnp.asarray(num_data * [[1, 0]])
x_2 = jax.random.normal(key_2, (num_data, 2)) + mu_2[None, :]
y_2 = jnp.asarray(num_data * [[0, 1]])
x_train = jnp.concatenate([x_1, x_2], axis=0)
y_train = jnp.concatenate([y_1, y_2], axis=0)

# Create model
model_init, model_apply = bnn_util.model_mlp(output_dimensions=2)
variables_dict = model_init(jax.random.PRNGKey(42), x_train)
variables, unflatten = jax.flatten_util.ravel_pytree(variables_dict)

# Optimise

optimizer = optax.adam(1e-2)
optimizer_state = optimizer.init(variables)
n_epochs = 200
key = jax.random.PRNGKey(412576)
batch_size = num_data


def loss_p(v, x, y):
    y_pred = model_apply(unflatten(v), x)
    return bnn_util.loss_training_cross_entropy(y_pred, y)


loss_value_and_grad = jax.jit(jax.value_and_grad(loss_p, argnums=0))

for epoch in range(n_epochs):
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
        acc = bnn_util.metric_accuracy(y_pred, y_train[idx])
        print(f"Epoch {epoch}, loss {loss:.3f}, accuracy {acc:.3f}")

# Optimize the calibration loss
log_alpha = 0.0
optimizer = optax.adam(1e-1)
optimizer_state = optimizer.init(log_alpha)

calib_loss = bnn_util.loss_calibration(
    model_fun=model_apply,
    param_unflatten=unflatten,
    loss_single=bnn_util.loss_training_cross_entropy_single,
    hyperparam_unconstrain=lambda s: 1e-3 + jnp.exp(s),
)
value_and_grad = jax.jit(jax.value_and_grad(calib_loss, argnums=0))

n_epochs = 200
for epoch in range(n_epochs):
    loss, grad = value_and_grad(log_alpha, variables, x_train, y_train)
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss {loss:.3f}, alpha {log_alpha:.3f}")

print("Successfull!!!!! :) :) :) ")
