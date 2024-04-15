"""Utilities for BNNs."""

import flax.linen
import jax
import jax.numpy as jnp


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


def loss_single(y_pred, y_data):
    logprobs = jax.nn.log_softmax(y_pred, axis=-1)
    return -jnp.sum(logprobs * y_data, axis=-1)


def accuracy(y_pred, y_data):
    return jnp.mean(jnp.argmax(y_pred, axis=-1) == jnp.argmax(y_data, axis=-1))


def calib_loss(model_fun, unflatten):
    def loss(a, variables, x_train, y_train):
        tmp1 = len(variables) / 2 * a
        tmp2 = -0.5 * _alpha(a) * jnp.dot(variables, variables)
        log_prior = tmp1 + tmp2

        M = ggn_fn(a, variables, x_train, y_train)
        _sign, logdet = jnp.linalg.slogdet(M)

        log_marginal = log_prior - 0.5 * logdet
        return -log_marginal

    def ggn_fn(a, variables, x_train, y_train):
        model_pred = model_fun(unflatten(variables), x_train)
        H = jax.vmap(jax.hessian(loss_single, argnums=0))(model_pred, y_train)
        J = jax.jacfwd(lambda v: model_fun(unflatten(v), x_train))(variables)
        ggn_summands = jax.vmap(lambda j, h: j.T @ h @ j)(J, H)
        return jnp.sum(ggn_summands, axis=0) + _alpha(a) * jnp.eye(J.shape[-1])

    def _alpha(a):
        return 1e-3 + jnp.exp(a)

    return loss
