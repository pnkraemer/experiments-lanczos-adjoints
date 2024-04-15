"""Utilities for BNNs."""

import flax.linen
import jax
import jax.numpy as jnp


def model_mlp(output_dimensions):
    class _MLP(flax.linen.Module):
        out_dims: int

        @flax.linen.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            x = flax.linen.Dense(8)(x)
            x = flax.linen.tanh(x)
            return flax.linen.Dense(self.out_dims)(x)

    model = _MLP(output_dimensions)
    return model.init, model.apply


def metric_accuracy(y_pred, y_data):
    return jnp.mean(jnp.argmax(y_pred, axis=-1) == jnp.argmax(y_data, axis=-1))


def loss_training_cross_entropy(y_pred, y_data):
    loss_single = jax.vmap(loss_training_cross_entropy_single)(y_pred, y_data)
    return jnp.mean(loss_single, axis=0)


def loss_training_cross_entropy_single(y_pred, y_data):
    logprobs = jax.nn.log_softmax(y_pred, axis=-1)
    return -jnp.sum(logprobs * y_data, axis=-1)


def loss_calibration(
    *, model_fun, param_unflatten, loss_single, hyperparam_unconstrain
):
    def loss(a, variables, x_train, y_train):
        alpha = hyperparam_unconstrain(a)
        tmp1 = len(variables) / 2 * a
        tmp2 = -0.5 * alpha * jnp.dot(variables, variables)
        log_prior = tmp1 + tmp2

        M = ggn_fn(a, variables, x_train, y_train)
        _sign, logdet = jnp.linalg.slogdet(M)

        log_marginal = log_prior - 0.5 * logdet
        return -log_marginal

    def ggn_fn(a, variables, x_train, y_train):
        alpha = hyperparam_unconstrain(a)
        model_pred = model_fun(param_unflatten(variables), x_train)

        H = jax.vmap(jax.hessian(loss_single, argnums=0))(model_pred, y_train)
        J = jax.jacfwd(lambda v: model_fun(param_unflatten(v), x_train))(variables)

        ggn_summands = jax.vmap(lambda j, h: j.T @ h @ j)(J, H)
        return jnp.sum(ggn_summands, axis=0) + alpha * jnp.eye(J.shape[-1])

    return loss
