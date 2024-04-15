# todo:
#  - Reconsider all loss functions
#  - Use them to evaluate the calibration result
#  - Plot solutions
#  - Move some code to the src
#  - Split the script into different smaller scripts once big


import os

import flax.linen
import jax
import jax.numpy as jnp
import optax
from matfree_extensions import exp_util


class MLP(flax.linen.Module):
    out_dims: int

    @flax.linen.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = flax.linen.Dense(8)(x)
        x = flax.linen.tanh(x)
        return flax.linen.Dense(self.out_dims)(x)


def f(x):
    return jnp.sin(x)


if __name__ == "__main__":
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
    model = MLP(out_dims=2)
    variables_dict = model.init(jax.random.PRNGKey(42), x_train)
    variables, unflatten = jax.flatten_util.ravel_pytree(variables_dict)

    # Loss and accuracy

    def loss_single(y_pred, y_data):
        logprobs = jax.nn.log_softmax(y_pred, axis=-1)
        return -jnp.sum(logprobs * y_data, axis=-1)

    def accuracy(params, x_, y_):
        y_pred = model.apply(unflatten(params), x_)
        return jnp.mean(jnp.argmax(y_pred, axis=-1) == jnp.argmax(y_, axis=-1))

    # Optimise

    optimizer = optax.adam(1e-2)
    optimizer_state = optimizer.init(variables)
    n_epochs = 200
    key = jax.random.PRNGKey(412576)
    batch_size = num_data

    def loss_p(v, x, y):
        return jnp.mean(jax.vmap(loss_single)(model.apply(unflatten(v), x), y), axis=0)

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
            acc = accuracy(variables, x_train[idx], y_train[idx])
            print(f"Epoch {epoch}, loss {loss:.3f}, accuracy {acc:.3f}")

    # Construct the GGN Matrix
    def calib_loss(log_alpha):
        D = len(variables)
        log_prior = D / 2 * log_alpha - 0.5 * _alpha(log_alpha) * jnp.dot(
            variables, variables
        )
        M = ggn_fn(log_alpha)
        _sign, logdet = jnp.linalg.slogdet(M)
        log_marginal = log_prior - 0.5 * logdet
        return -log_marginal

    def ggn_fn(log_alpha):
        model_pred = model.apply(unflatten(variables), x_train)
        H = jax.vmap(jax.hessian(loss_single, argnums=0))(model_pred, y_train)
        J = jax.jacfwd(lambda v: model.apply(unflatten(v), x_train))(variables)
        ggn_summands = jax.vmap(lambda j, h: j.T @ h @ j)(J, H)
        return jnp.sum(ggn_summands, axis=0) + _alpha(log_alpha) * jnp.eye(J.shape[-1])

    def _alpha(log_alpha):
        return 1e-3 + jnp.exp(log_alpha)

    # Optimize the calibration loss
    log_alpha = 0.0
    optimizer = optax.adam(1e-1)
    optimizer_state = optimizer.init(log_alpha)
    value_and_grad = jax.jit(jax.value_and_grad(calib_loss))
    n_epochs = 200
    for epoch in range(n_epochs):
        loss, grad = value_and_grad(log_alpha)
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        log_alpha = optax.apply_updates(log_alpha, updates)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss {loss:.3f}, alpha {_alpha(log_alpha):.3f}")

    print("Successfull!!!!! :) :) :) ")
