"""Utilities for BNNs."""

from typing import Callable

import flax.linen
import jax
import jax.numpy as jnp
from matfree import hutchinson

from matfree_extensions import lanczos


def model_mlp(*, out_dims, activation: Callable):
    class _MLP(flax.linen.Module):
        out_dims: int
        activation: Callable

        @flax.linen.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            x = flax.linen.Dense(5)(x)
            x = self.activation(x)
            x = flax.linen.Dense(5)(x)
            x = self.activation(x)
            x = flax.linen.Dense(5)(x)
            x = self.activation(x)
            x = flax.linen.Dense(5)(x)
            x = self.activation(x)
            return flax.linen.Dense(self.out_dims)(x)

    model = _MLP(out_dims=out_dims, activation=activation)
    return model.init, model.apply


def metric_accuracy(y_pred, y_data):
    return jnp.mean(jnp.argmax(y_pred, axis=-1) == jnp.argmax(y_data, axis=-1))


def loss_training_cross_entropy(y_pred, y_data):
    loss_single = jax.vmap(loss_training_cross_entropy_single)(y_pred, y_data)
    return jnp.mean(loss_single, axis=0)


def loss_training_cross_entropy_single(y_pred, y_data):
    logprobs = jax.nn.log_softmax(y_pred, axis=-1)
    return -jnp.sum(logprobs * y_data, axis=-1)


def loss_calibration(*, ggn_fun, hyperparam_unconstrain, logdet_fun):
    def loss(a, variables, x_train, y_train, *logdet_params):
        alpha = hyperparam_unconstrain(a)
        tmp1 = len(variables) / 2 * a
        tmp2 = -0.5 * alpha * jnp.dot(variables, variables)
        log_prior = tmp1 + tmp2

        M = ggn_fun(alpha, variables, x_train, y_train)
        logdet = logdet_fun(M, *logdet_params)

        log_marginal = log_prior - 0.5 * logdet
        return -log_marginal

    return loss


def solver_logdet_dense():
    def logdet(M: jax.Array):
        _sign, logdet_value = jnp.linalg.slogdet(M)
        return logdet_value

    return logdet


# ~10 because the rank of our GGN is 10
def solver_logdet_sparse(*, slq_rank=10, slq_num_samples=100, slq_num_batches=1):
    def logdet(M: jax.Array, key: jax.random.PRNGKey):
        x_like = jnp.ones((len(M),), dtype=float)
        sampler = hutchinson.sampler_rademacher(x_like, num=slq_num_samples)

        integrand = lanczos.integrand_spd(jnp.log, slq_rank, lambda v: M @ v)
        estimate = hutchinson.hutchinson(integrand, sampler)

        keys = jax.random.split(key, num=slq_num_batches)
        values = jax.lax.map(lambda k: estimate(k), keys)
        return jnp.mean(values, axis=0)

    return logdet


def predictive_variance(*, ggn_fun, model_fun, param_unflatten, hyperparam_unconstrain):
    def evaluate(a, variables, x_train, y_train, x_test):
        alpha = hyperparam_unconstrain(a)
        ggn = ggn_fun(alpha, variables, x_train, y_train)

        covariance = jnp.linalg.inv(ggn)
        J_test = jax.jacfwd(lambda v: model_fun(param_unflatten(v), x_test))(variables)
        return jax.vmap(lambda J_single: J_single @ covariance @ J_single.T)(J_test)

    return evaluate


def ggn(*, loss_single, model_fun, param_unflatten):
    def ggn_fun(alpha, variables, x_train, y_train):
        model_pred = model_fun(param_unflatten(variables), x_train)

        H = jax.vmap(jax.hessian(loss_single, argnums=0))(model_pred, y_train)
        J = jax.jacfwd(lambda v: model_fun(param_unflatten(v), x_train))(variables)

        ggn_summands = jax.vmap(lambda j, h: j.T @ h @ j)(J, H)
        return jnp.sum(ggn_summands, axis=0) + alpha * jnp.eye(J.shape[-1])

    return ggn_fun
