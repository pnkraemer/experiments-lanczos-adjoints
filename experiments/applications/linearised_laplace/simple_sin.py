import os

import flax.linen
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
    x_train = jnp.linspace(0, 2 * jnp.pi, num_data)
    eps = jax.random.normal(jax.random.PRNGKey(0), (num_data,)) * 0.05
    y_train = f(x_train) + eps

    # Plot data
    plt.plot(x_train, y_train, "o")
    plt.savefig(f"{directory}/figure.pdf")

    # Create model
    model = MLP(out_dims=1)
    variables_dict = model.init(jax.random.PRNGKey(42), x_train)
    variables, unflatten = jax.flatten_util.ravel_pytree(variables_dict)

    def apply_fn(params: jax.Array, x: jax.Array):
        return model.apply(unflatten(params), x)

    y = apply_fn(variables, x_train)

    # Train model via optax
    def loss_fn(params, x_, y_):
        y_pred = apply_fn(params, x_).reshape((-1,))
        return jnp.mean((y_pred - y_) ** 2, axis=0)

    optimizer = optax.sgd(1e-1)
    optimizer_state = optimizer.init(variables)
    n_epochs = 200
    key = jax.random.PRNGKey(412576)
    batch_size = num_data
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    for epoch in range(n_epochs):
        # Subsample data
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, x_train.shape[0], (batch_size,), replace=False)

        loss, grad = value_and_grad(variables, x_train[idx], y_train[idx])
        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        variables = optax.apply_updates(variables, updates)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss {loss:.3f}")

    # Plot predictions
    x_val = jnp.linspace(0, 2 * jnp.pi, 1000)
    y_pred = apply_fn(variables, x_val)
    plt.plot(x_val, f(x_val))
    plt.plot(x_val, y_pred)
    plt.savefig(f"{directory}/predixcyions.pdf")

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

    def data_likelihood(log_alpha, x_test, y_test):
        # GP in weight space
        mean = variables
        cov = jnp.linalg.inv(ggn_fn(log_alpha))

        # GP in y space
        mean_fn = (apply_fn)(mean, x_test).reshape((-1,))
        J_test = (jax.jacfwd(lambda p: apply_fn(p, x_test))(variables)).squeeze()
        cov_fn = J_test @ cov @ J_test.T

        # Compute the likelihood
        cov_matrix = (cov_fn + cov_fn.T) / 2 + jnp.eye(100) * 1e-6
        return jax.scipy.stats.multivariate_normal.logpdf(y_test, mean_fn, cov_matrix)

    def ggn_fn(log_alpha):
        J = (jax.jacfwd(lambda p: apply_fn(p, x_train))(variables)).squeeze()
        GGN = J.T @ J
        return GGN + _alpha(log_alpha) * jnp.eye(GGN.shape[0])

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

    print("Log Likelihood (before):", data_likelihood(0.0, x_train, y_train))
    print("Log Likelihood (after):", data_likelihood(log_alpha, x_train, y_train))
