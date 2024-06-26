"""Utilities for BNNs."""

import functools
from typing import Callable

import flax.linen
import jax
import jax.numpy as jnp
from jax._src import linear_util as lu
from jax._src.util import wraps
from matfree import hutchinson
from tqdm import tqdm

from matfree_extensions import lanczos
from matfree_extensions.util import gp_util_linalg
from matfree_extensions.util.bnn_baselines import hutchinson_diagonal

# TODO: Decide if we abbreviate metric in function names


def model_mlp(*, out_dims, activation: Callable):
    class _MLP(flax.linen.Module):
        out_dims: int
        activation: Callable

        @flax.linen.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            x = flax.linen.Dense(50)(x)
            x = self.activation(x)
            x = flax.linen.Dense(50)(x)
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


def loss_training_cross_entropy(logits, labels_hot):
    loss_vmapped = jax.vmap(loss_training_cross_entropy_single)
    loss_single = loss_vmapped(logits, labels_hot)
    return jnp.mean(loss_single, axis=0)


def loss_training_cross_entropy_single(logits, labels_hot):
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(logprobs * labels_hot, axis=-1)


# todo: move to gp? (And rename gp.py appropriately, of course)
#  laplace-torch calls this Laplace.log_prob(normalized=True)
# todo: rename unconstrain() to constrain()
def loss_calibration(*, ggn_fun, hyperparam_unconstrain, logdet_fun):
    def loss(a, variables, x_train, y_train, *logdet_params):
        alpha = hyperparam_unconstrain(a)

        tmp1 = len(variables) / 2 * jnp.log(alpha)

        tmp2 = -0.5 * alpha * jnp.dot(variables, variables)
        log_prior = tmp1 + tmp2

        M = ggn_fun(alpha, variables, x_train, y_train)
        logdet = logdet_fun(M, *logdet_params)

        log_marginal = log_prior - 0.5 * logdet
        return -log_marginal

    return loss


# todo: move to gp? (And rename gp.py appropriately, of course)
#  laplace-torch calls this Laplace.log_prob(normalized=True)
def loss_log_prob_like_in_redux(*, ggn_fun, hyperparam_unconstrain, logdet_fun):
    def loss(a, variables, x_train, y_train, *logdet_params):
        alpha = hyperparam_unconstrain(a)

        M = ggn_fun(alpha, variables, x_train, y_train)
        logdet = logdet_fun(M, *logdet_params)

        tmp1 = -len(variables) / 2 * jnp.log(2 * jnp.pi) + logdet / 2
        tmp2 = -jnp.dot(variables, variables) / 2
        return tmp1 + tmp2

    return loss


# todo: move to gp? (And rename gp.py appropriately, of course)
def solver_logdet_dense():
    def logdet(M: jax.Array):
        _sign, logdet_value = jnp.linalg.slogdet(M)
        return logdet_value

    return logdet


def slq_log_clipped(*, clip_value=1.0):
    def log(x):
        eps = jnp.finfo(x.dtype).eps
        x_clipped = jnp.where(x < eps, clip_value, x)
        return jnp.log(x_clipped)

    return log


# todo: move to gp? (And rename gp.py appropriately, of course)
def solver_logdet_slq(*, lanczos_rank, slq_num_samples, slq_num_batches):
    def logdet(M: jax.Array, key: jax.random.PRNGKey):
        x_like = jnp.ones((len(M),), dtype=float)
        sampler = hutchinson.sampler_rademacher(x_like, num=slq_num_samples)

        matfun = slq_log_clipped()
        integrand = lanczos.integrand_spd(matfun, lanczos_rank, lambda v: M @ v)
        estimate = hutchinson.hutchinson(integrand, sampler)

        keys = jax.random.split(key, num=slq_num_batches)
        values = jax.lax.map(lambda k: estimate(k), keys)
        return jnp.mean(values, axis=0)

    return logdet


def solver_logdet_slq_implicit(
    *, lanczos_rank, slq_num_samples, slq_num_batches, N: int
):
    x_like = jnp.ones((N,), dtype=float)
    sampler = hutchinson.sampler_rademacher(x_like, num=slq_num_samples)

    def logdet(Av: Callable, key: jax.random.PRNGKey, *args):
        integrand = lanczos.integrand_spd(jnp.log, lanczos_rank, Av)
        estimate = hutchinson.hutchinson(integrand, sampler)

        keys = jax.random.split(key, num=slq_num_batches)
        values = jax.lax.map(lambda k: estimate(k, *args), keys)
        return jnp.mean(values, axis=0)

    return logdet


def predictive_cov(*, ggn_fun, model_fun, param_unflatten, hyperparam_unconstrain):
    def evaluate(a, variables, x_train, y_train, x_test):
        alpha = hyperparam_unconstrain(a)
        ggn = ggn_fun(alpha, variables, x_train, y_train)

        covariance = jnp.linalg.inv(ggn)
        J_test = jax.jacfwd(lambda v: model_fun(param_unflatten(v), x_test))(variables)
        return jax.vmap(lambda J_single: J_single @ covariance @ J_single.T)(J_test)

    return evaluate


def ggn_full(*, loss_single, model_fun, param_unflatten):
    def ggn_fun(alpha, variables, x_train, y_train):
        model_pred = model_fun(param_unflatten(variables), x_train)

        H = jax.vmap(jax.hessian(loss_single, argnums=0))(model_pred, y_train)
        J = jax.jacfwd(lambda v: model_fun(param_unflatten(v), x_train))(variables)

        ggn_summands = jax.vmap(lambda j, h: j.T @ h @ j)(J, H)
        return jnp.sum(ggn_summands, axis=0) + alpha * jnp.eye(J.shape[-1])

    return ggn_fun


def ggn_vp_running(*, loss_single, model_fun, param_unflatten):
    def gvp(v_vec, params_vec, x_batch, y_batch):
        # v_like_params = param_unflatten(v_vec)
        params = param_unflatten(params_vec)

        # def model_flat(p, x):
        #     return model_fun(param_unflatten(p), x)
        def scan_fun(carry, batch):
            x, y = batch
            x = x[None, ...]
            y = y[None, ...]

            def model_pred(p):
                return model_fun(p, x)
                # return model_flat(p, x)

            # Big models(except ConvNext) return a tuple (logits, model_state)
            preds, Jv = jax.jvp(model_pred, (params_vec,), (v_vec,))
            _, vjp_fn = jax.vjp(model_pred, params_vec)
            H = jax.vmap(jax.hessian(loss_single, argnums=0))(preds, y)
            HJv = jnp.einsum("boi, bi->bo", H, Jv)
            JtHJv = vjp_fn(HJv)[0]
            # return carry + JtHJv, None
            return jax.tree_map(lambda c, v: c + v, carry, JtHJv), None

        init_value = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        # init_value = jnp.zeros_like(params_vec)
        return jax.lax.scan(scan_fun, init_value, (x_batch, y_batch))[0]

    return gvp


def ggn_vp_parallel(*, loss_single, model_fun, param_unflatten):
    def gvp(v_vec, params_vec, x_batch, y_batch):
        v_like_params = param_unflatten(v_vec)
        params = param_unflatten(params_vec)

        def body_fn(x_single, y_single):
            x = x_single[None, ...]
            y = y_single[None, ...]

            def model_pred(p):
                return model_fun(p, x)

            # Big models(except ConvNext) return a tuple (logits, model_state)
            # Hence: model_pred = lambda p: model_fun(p, x)[0]

            # preds, Jv = jax.jvp(model_pred, (params,), (v_like_params,))
            # _, vjp_fn = jax.vjp(model_pred, params)

            preds, jvp_fn = jax.linearize(model_pred, params)
            vjp_fn = jax.linear_transpose(jvp_fn, params)

            Jv = jvp_fn(v_like_params)
            H = jax.vmap(jax.hessian(loss_single, argnums=0))(preds, y)
            HJv = jnp.einsum("boi, bi->bo", H, Jv)
            return vjp_fn(HJv)[0]

        return jax.tree_map(
            lambda x: x.sum(axis=0), jax.vmap(body_fn)(x_batch, y_batch)
        )

    return gvp


def kernel_vp_parallel(*, loss_single, model_fun, param_unflatten):
    def gvp(v_like_outs, params_vec, x_batch, y_batch):
        v_like_outs = v_like_outs[None, ...]
        params = param_unflatten(params_vec)

        def body_fn(x_single, y_single):
            x = x_single[None, ...]
            y = y_single[None, ...]

            def model_pred(p):
                return model_fun(p, x)

            # Big models(except ConvNext) return a tuple (logits, model_state)
            # Hence: model_pred = lambda p: model_fun(p, x)[0]
            preds, vjp_fn = jax.vjp(model_pred, params)
            H = jax.vmap(jax.hessian(loss_single, argnums=0))(preds, y)
            H_sqrt = jnp.linalg.cholesky(H)
            Hv = jnp.einsum("boi, bi->bo", H_sqrt, v_like_outs)
            JtHv = vjp_fn(Hv)[0]
            _, JJtHv = jax.jvp(model_pred, (params,), (JtHv,))
            return jnp.einsum("boi, bi->bo", H, JJtHv)

        return jax.tree_map(
            lambda x: x.sum(axis=0), jax.vmap(body_fn)(x_batch, y_batch)
        )

    return gvp


def ggn_vp_dataloader(
    param_vec, loss_single, model_fun, param_unflatten, data_loader, sum_type="parallel"
):
    def ggn_vec_prod(v_vec):
        if sum_type == "parallel":
            ggn_vp = ggn_vp_parallel
        elif sum_type == "running":
            ggn_vp = ggn_vp_running
        ggn_vp_fn = ggn_vp(
            model_fun=model_fun,
            loss_single=loss_single,
            param_unflatten=param_unflatten,
        )
        ggn_vp_fn = jax.jit(ggn_vp_fn)
        gvp = jnp.zeros_like(param_vec)
        for _, batch in enumerate(tqdm(data_loader)):
            x_batch, y_batch = batch["image"], batch["label"]
            gvp_tree = ggn_vp_fn(v_vec, param_vec, x_batch, y_batch)
            gvp += jax.flatten_util.ravel_pytree(gvp_tree)[0]
        return gvp

    return ggn_vec_prod


def ggn_diag(*, loss_single, model_fun, param_unflatten):
    ggn_fun_full = ggn_full(
        loss_single=loss_single, model_fun=model_fun, param_unflatten=param_unflatten
    )

    def ggn_fun(alpha, variables, x_train, y_train):
        ggn = ggn_fun_full(alpha, variables, x_train, y_train)
        return jnp.diag(jnp.diag(ggn))

    return ggn_fun


def sampler_cholesky(*, ggn_fun, num):
    def sample(key, alpha, variables, x_train, y_train):
        ggn = ggn_fun(alpha, variables, x_train, y_train)
        ggn_inv_sqrt = jnp.linalg.cholesky(jnp.linalg.inv(ggn))

        eps = jax.random.normal(key, (num, *variables.shape))
        return jnp.dot(ggn_inv_sqrt, eps.T).T + variables[None, ...]

    return sample


def sampler_lanczos(*, ggn_fun, num, lanczos_rank):
    def sample(key, alpha, variables, x_train, y_train):
        ggn = ggn_fun(alpha, variables, x_train, y_train)

        tridiag = lanczos.tridiag(lambda v: ggn @ v, lanczos_rank, reortho="full")
        eps = jax.random.normal(key, (num, *variables.shape))

        sample_one = functools.partial(_sample_single, tridiag=tridiag)
        return jax.vmap(sample_one)(eps) + variables[None, ...]

    def _sample_single(eps, *, tridiag):
        (Q, tridiag), _ = tridiag(eps)
        dense_matrix = _dense_tridiag(*tridiag)

        tri_inv_sqrt = jnp.linalg.cholesky(jnp.linalg.inv(dense_matrix))
        return Q.T @ (tri_inv_sqrt @ (Q @ eps))

    return sample


def lanczos_sampler(*, ggn_vp, num_samples, lanczos_rank, key, params_vec):
    eps = jax.random.normal(key, (num_samples, *params_vec.shape))

    def posterior_sampler(single_sample):
        tridiag = lanczos.tridiag(ggn_vp, lanczos_rank, reortho="full")
        (Q, tridiag), _ = tridiag(single_sample)
        dense_matrix = _dense_tridiag(*tridiag)
        (w, v) = jnp.linalg.eigh(dense_matrix)
        eigvecs = Q.T @ v
        eigvals = w
        eigvals = jnp.where(w < 1e-9, 1.0, eigvals)
        inv_eigvals = 1 / eigvals
        inv_eigvals = jnp.where(w < 1e-9, 0.0, inv_eigvals)
        sample = (jnp.sqrt(inv_eigvals)) * single_sample.at[:lanczos_rank].get()
        sample = params_vec + (eigvecs @ sample)
        return sample

    return jax.vmap(posterior_sampler)(eps)


def _dense_tridiag(diagonal, off_diagonal):
    return jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, H', W', p_H, p_W, C]
    x = x.reshape(B, -1, *x.shape[3:])  # [B, H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.reshape(B, x.shape[1], -1)  # [B, H'*W', p_H*p_W*C]
    return x


def callibration_loss_diagonal(
    model_apply,
    unflatten,
    hyperparam_unconstrain,
    hutchinson_samples,
    num_levels,
    n_params,
):
    # get_diag_fn = functools.partial(
    #     exact_diagonal,
    #     model_fn=model_apply,
    #     output_dim=num_classes,
    #     likelihood="classification",
    # )
    gvp_fn = ggn_vp_parallel(
        loss_single=loss_training_cross_entropy_single,
        model_fun=model_apply,
        param_unflatten=unflatten,
    )
    key = jax.random.PRNGKey(0)
    get_diag_fn = functools.partial(
        hutchinson_diagonal,
        n_samples=hutchinson_samples,
        key=key,
        computation_type="serial",
        num_levels=num_levels,
    )

    def loss(log_alpha, params_vec, img, label):
        alpha = hyperparam_unconstrain(log_alpha)
        gvp_fn_batch = jax.tree_util.Partial(
            gvp_fn, params_vec=params_vec, x_batch=img, y_batch=label
        )
        diag = get_diag_fn(gvp_fn=gvp_fn_batch, params=unflatten(params_vec))
        diag_vec = jax.flatten_util.ravel_pytree(diag)[0]
        diag_vec = jnp.where(diag_vec < 1e-4, 0.0, diag_vec)
        logdet = jnp.sum(jnp.log(diag_vec + alpha))
        log_prior = jnp.log(alpha) * n_params - alpha * jnp.dot(params_vec, params_vec)
        log_marginal = log_prior - logdet
        return -log_marginal

    return loss


def callibration_loss(model_apply, unflatten, hyperparam_unconstrain, n_params):
    ggn_fun = ggn_vp_parallel(
        loss_single=loss_training_cross_entropy_single,
        model_fun=model_apply,
        param_unflatten=unflatten,
    )

    def ggn_mat(v_vec, alpha, *params):
        Gv_tree = ggn_fun(v_vec, *params)
        return jax.flatten_util.ravel_pytree(Gv_tree)[0] + alpha * v_vec

    def loss(log_alpha, params_vec, img, label, key):
        alpha = hyperparam_unconstrain(log_alpha)
        logdet_fun = solver_logdet_slq_implicit(
            lanczos_rank=10, slq_num_samples=10, slq_num_batches=1, N=n_params
        )

        logdet = logdet_fun(ggn_mat, key, alpha, params_vec, img, label)
        log_prior = jnp.log(alpha) * n_params - alpha * jnp.dot(params_vec, params_vec)
        log_marginal = log_prior - logdet
        return -log_marginal

    return loss


def vectorize_nn(model_fn, params):
    """Vectorize the Neural Network
    Inputs:
    parameters: Pytree of parameters
    model_fn: A function that takes in pytree parameters and data

    Outputs:
    params_vec: Vectorized parameters
    unflatten_fn: Unflatten function
    model_apply_vec: A function that takes in vectorized parameters and data
    """
    params_vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    def model_apply_vec(params_vectorized, x):
        return model_fn(unflatten_fn(params_vectorized), x)

    return params_vec, unflatten_fn, model_apply_vec


def get_model_apply_fn(model_name, model_apply, batch_stats=None, rng=None):
    if model_name in ["ResNet_small", "ResNet18", "DenseNet", "GoogleNet"]:
        assert (
            batch_stats is not None
        ), "Batch statistics must be provided for ResNet and DenseNet models."

        def model_fn(params, imgs):
            return model_apply(
                {"params": params, "batch_stats": batch_stats},
                imgs,
                train=False,
                mutable=False,
            )
    elif model_name in ["LeNet", "MLP"]:
        model_fn = model_apply
    elif model_name == "VisionTransformer":
        assert rng is not None, "RNG key must be provided for Vision Transformer model."

        def model_fn(params, imgs):
            return model_apply(
                {"params": params}, imgs, train=False, rngs={"dropout": rng}
            )
    else:
        raise ValueError

    return model_fn


def logpdf_cholesky() -> Callable:
    """Construct a logpdf function that relies on a Cholesky decomposition."""

    def logpdf(y, /, *, mean, cov: Callable):
        # Materialise the covariance matrix
        cov_matrix = jax.jacfwd(cov)(mean)

        # Cholesky-decompose
        cholesky = jnp.linalg.cholesky(cov_matrix)

        # Log-determinant
        logdet = jnp.sum(jnp.log(jnp.diag(cholesky)))
        # Mahalanobis norm

        def solve_triangular(A, b):
            return jax.scipy.linalg.solve_triangular(A, b, lower=True, trans=False)

        tmp = solve_triangular(cholesky, y - mean)
        mahalanobis = jnp.dot(tmp, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)

        return -logdet - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), {}

    return logpdf


from jax._src.api import _jacfwd_unravel, _jvp, _std_basis
from jax._src.api_util import argnums_partial


def jacfwd_map(fun: Callable, argnums: int = 0) -> Callable:
    @wraps(fun, argnums=argnums)
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        pushfwd: Callable = functools.partial(_jvp, f_partial, dyn_args)
        y, jac = jax.lax.map(pushfwd, _std_basis(dyn_args))
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = jax.tree_map(
            functools.partial(_jacfwd_unravel, example_args), y, jac
        )
        return jac_tree

    return jacfun


def logpdf_eigh() -> Callable:
    """Construct a logpdf function that uses a symmetric eigendecomposition."""

    def logpdf(y, /, *, mean, cov: Callable):
        # Materialise the covariance matrix
        # cov_matrix = jax.jacfwd(cov)(mean)
        cov_matrix = jacfwd_map(cov)(mean)

        # Compute H^(1/2) via eigh()
        (w, v) = jnp.linalg.eigh(cov_matrix)
        # w = jnp.maximum(0.0, w)  # clamp to allow sqrts

        # Log-determinant
        w_ = jnp.where(w < 1e-6, 1.0, w)
        logdet = jnp.sum(jnp.log(w_)) / 2

        # Mahalanobis norm
        inv_eigvals = 1 / w
        inv_eigvals = jnp.where(w < 1e-6, 0.0, inv_eigvals)
        factor = (v * jnp.sqrt(inv_eigvals[..., None, :])) @ v.T
        tmp = factor @ (y - mean)
        mahalanobis = jnp.dot(tmp, tmp)

        # Combine the terms
        (n,) = jnp.shape(mean)

        return -logdet - 0.5 * mahalanobis - n / 2 * jnp.log(2 * jnp.pi), {}

    return logpdf


def predictive_posterior_loglikelihood(*, model_apply, unflatten, logpdf, ggn_fun):
    def eval_logprob(params_vec, x_test, y_test):
        # p(y| x, theta) = N(y | f(x, theta), J_* H^-1 J_*^T)
        mean_pred, jvp_fn = jax.linearize(
            lambda p: model_apply(unflatten(p), x_test), params_vec
        )
        vjp_fn = jax.linear_transpose(jvp_fn, params_vec)
        y_flat, unflat = jax.flatten_util.ravel_pytree(y_test)
        mean_flat, _unflat = jax.flatten_util.ravel_pytree(mean_pred)

        def cov_vp(v_):
            v = unflat(v_)
            (Jtv,) = vjp_fn(v)
            cg = gp_util_linalg.krylov_solve_cg_fixed_step_reortho(20)
            # inv_ggn, _info = matfree.gp_utils.krylov_solve_cg_fixed_step_reortho()
            # jax.scipy.sparse.linalg.cg(ggn_fun, Jtv, tol=0.1)
            inv_ggn, _info = cg(ggn_fun, Jtv)
            out = jvp_fn(inv_ggn)
            return jax.flatten_util.ravel_pytree(out)[0]

        return logpdf(y_flat, mean=mean_flat, cov=cov_vp)

    return eval_logprob


def predictive_logit_sampler(*, model_apply, unflatten, num_samples, ggn_fun):
    def eval_test_set(params_vec, x_test, y_test, key):
        mean_pred, jvp_fn = jax.linearize(
            lambda p: model_apply(unflatten(p), x_test), params_vec
        )

        vjp_fn = jax.linear_transpose(jvp_fn, params_vec)
        y_flat, unflat = jax.flatten_util.ravel_pytree(y_test)
        mean_flat, _unflat = jax.flatten_util.ravel_pytree(mean_pred)

        def cov_vp(v_):
            v = unflat(v_)
            (Jtv,) = vjp_fn(v)
            cg = gp_util_linalg.krylov_solve_cg_fixed_step_reortho(20)
            inv_ggn, _info = cg(ggn_fun, Jtv)
            out = jvp_fn(inv_ggn)
            return jax.flatten_util.ravel_pytree(out)[0]

        cov_matrix = jacfwd_map(cov_vp)(mean_flat)
        (w, v) = jnp.linalg.eigh(cov_matrix)
        # Mahalanobis norm
        inv_eigvals = 1 / w
        inv_eigvals = jnp.where(w < 1e-6, 0.0, inv_eigvals)
        cov_sqrt = (v * jnp.sqrt(inv_eigvals[..., None, :])) @ v.T
        eps = jax.random.normal(key, (num_samples, *mean_flat.shape))
        samples = jax.vmap(lambda e: mean_flat + cov_sqrt @ e)(eps)
        return jax.vmap(unflat)(samples)

    return eval_test_set
