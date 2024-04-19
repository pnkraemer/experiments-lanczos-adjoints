"""Utilities for BNNs."""

from typing import Callable

import flax.linen
import jax
import jax.numpy as jnp
from matfree import hutchinson

from matfree_extensions import lanczos

# TODO: Decide if we abbreviate metric in function names


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


def metric_accuracy(*, probs, labels_hot):
    assert probs.ndim == 2
    assert labels_hot.ndim == 2
    acc = jnp.argmax(probs, axis=-1) == jnp.argmax(labels_hot, axis=-1)
    return jnp.mean(acc, axis=-1)


def metric_nll(*, logits, labels_hot, sum_or_mean_fun=jnp.sum):
    assert logits.ndim == 2
    assert labels_hot.ndim == 2
    logprobs = jax.nn.log_softmax(logits)
    nll = jnp.sum(labels_hot * logprobs, axis=-1)
    return -sum_or_mean_fun(nll, axis=0)


def metric_confidence(*, probs):
    assert probs.ndim == 2
    confs = jnp.max(probs, axis=-1)
    return jnp.mean(confs, axis=0)


# def metric_auroc(*, labels_pred, labels_hot):
#     return sklern
#     def get_auroc(py_in, py_out):
#         labels = np.zeros(len(py_in) + len(py_out), dtype='int32')
#         labels[:len(py_in)] = 1
#         examples = np.concatenate([py_in.max(1), py_out.max(1)])
#         return roc_auc_score(labels, examples).item()
#


def metric_ece(*, probs, labels_hot, num_bins):
    # Put the confidence into M bins
    _, bins = jnp.histogram(probs, bins=num_bins, range=(0, 1))

    preds = probs.argmax(axis=1)
    labels = labels_hot.argmax(axis=1)  # This line?
    confs = jnp.max(probs, axis=1)
    conf_idxs = jnp.digitize(confs, bins=bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(num_bins):
        preds_i = preds[conf_idxs == i]
        labels_i = labels[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = jnp.mean(preds_i == labels_i)
        conf = jnp.mean(confs_i)

        # Todo: think about handling empty bins
        # Mean of empty array defaults to NaN, but it should be zero
        if not jnp.isnan(acc) and not jnp.isnan(conf):
            accs_bin.append(acc)
            confs_bin.append(conf)
            nitems_bin.append(len(preds_i))

    accs_bin = jnp.asarray(accs_bin)
    confs_bin = jnp.asarray(confs_bin)
    nitems_bin = jnp.asarray(nitems_bin)

    ce = jnp.abs(confs_bin - accs_bin)
    weights = nitems_bin / nitems_bin.sum()

    # avg() not mean() because weights
    ce_avg_weighted = jnp.average(ce, weights=weights)
    ce_max = jnp.max(ce)
    return ce_avg_weighted, ce_max


def loss_training_cross_entropy(y_pred, y_data):
    loss_single = jax.vmap(loss_training_cross_entropy_single)(y_pred, y_data)
    return jnp.mean(loss_single, axis=0)


def loss_training_cross_entropy_single(y_pred, y_data):
    logprobs = jax.nn.log_softmax(y_pred, axis=-1)
    return -jnp.sum(logprobs * y_data, axis=-1)


# todo: move to gp? (And rename gp.py appropriately, of course)
#  laplace-torch calls this Laplace.log_prob(normalized=True)
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


# todo: move to gp? (And rename gp.py appropriately, of course)
def solver_logdet_dense():
    def logdet(M: jax.Array):
        _sign, logdet_value = jnp.linalg.slogdet(M)
        return logdet_value

    return logdet


# todo: move to gp? (And rename gp.py appropriately, of course)
def solver_logdet_slq(*, lanczos_rank, slq_num_samples, slq_num_batches):
    def logdet(M: jax.Array, key: jax.random.PRNGKey):
        x_like = jnp.ones((len(M),), dtype=float)
        sampler = hutchinson.sampler_rademacher(x_like, num=slq_num_samples)

        integrand = lanczos.integrand_spd(jnp.log, lanczos_rank, lambda v: M @ v)
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


def ggn_diag(*, loss_single, model_fun, param_unflatten):
    ggn_fun_dense = ggn(
        loss_single=loss_single, model_fun=model_fun, param_unflatten=param_unflatten
    )

    def ggn_fun(alpha, variables, x_train, y_train):
        ggn = ggn_fun_dense(alpha, variables, x_train, y_train)
        return jnp.diag(jnp.diag(ggn))

    return ggn_fun


def sampler_cholesky(ggn_fun, num):
    def sample(key, alpha, variables, x_train, y_train):
        GGN = ggn_fun(alpha, variables, x_train, y_train)
        GGN_inv_sqrt = jnp.linalg.cholesky(jnp.linalg.inv(GGN))

        eps = jax.random.normal(key, (num, *variables.shape))
        return jnp.dot(GGN_inv_sqrt, eps.T).T + variables[None, ...]

    return sample
