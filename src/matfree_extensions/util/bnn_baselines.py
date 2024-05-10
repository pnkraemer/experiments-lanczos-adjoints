from functools import partial
from typing import Literal, Optional

import jax
import tree_math as tm
from jax import numpy as jnp


@partial(jax.jit, static_argnames=("model_fn", "output_dim", "likelihood"))
def exact_diagonal(
    model_fn,
    params,
    output_dim,
    x_train_batch,
    likelihood: Literal["classification", "regression"] = "classification",
):
    """ "
    This function computes the exact diagonal of the GGN matrix.
    """
    n_data_pts = x_train_batch.shape[0]
    output_dim_vec = jnp.arange(output_dim)
    diag_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)

    if likelihood == "regression":

        def body_fn(n, res):
            def single_dim_grad(carry, output_dim):
                def model_single_dim(p):
                    return model_fn(p, x_train_batch[n])[output_dim][0]

                # model_single_dim = lambda p: (
                #     model_fn(p, x_train_batch[n])[output_dim]
                # )[0]
                new_grad = jax.grad(model_single_dim)(params)
                out = jax.tree_map(lambda x, y: x + y**2, carry, new_grad)
                return out, None

            scan_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
            grad, _ = jax.lax.scan(single_dim_grad, scan_init, output_dim_vec)
            return jax.tree_map(lambda x, y: x + y, res, grad)

        return jax.lax.fori_loop(0, n_data_pts, body_fn, diag_init)

    if likelihood == "classification":
        output_dim_vec = jnp.arange(output_dim)
        grid = jnp.meshgrid(output_dim_vec, output_dim_vec)
        coord_list = [entry.ravel() for entry in grid]
        output_cross_vec = jnp.vstack(coord_list).T

        def body_fn(n, res):
            preds_i = model_fn(params, x_train_batch[n][None, ...])
            preds_i = jax.nn.softmax(preds_i, axis=1)
            preds_i = jax.lax.stop_gradient(preds_i)
            D = jax.vmap(jnp.diag)(preds_i)
            H = jnp.einsum("bo, bi->boi", preds_i, preds_i)
            H = D - H

            def single_dim_grad(carry, output_dims):
                o_1, o_2 = output_dims

                def model_single_dim_1(p):
                    return model_fn(p, x_train_batch[n][None, ...])[0, o_1]

                def model_single_dim_2(p):
                    return model_fn(p, x_train_batch[n][None, ...])[0, o_2]

                new_grad_1 = jax.grad(model_single_dim_1)(params)
                new_grad_2 = jax.grad(model_single_dim_2)(params)
                h = H.at[0, o_1, o_2].get()
                prod_grad = (tm.Vector(new_grad_1) * tm.Vector(new_grad_2)).tree
                out = jax.tree_map(lambda x, y: x + h * y, carry, prod_grad)
                return out, None

            scan_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
            grad, _ = jax.lax.scan(single_dim_grad, scan_init, output_cross_vec)
            del H
            return jax.tree_map(lambda x, y: x + y, res, grad)

        return jax.lax.fori_loop(0, n_data_pts, body_fn, diag_init)

    msg = f"Argument likelihood = {likelihood} unknown."
    raise ValueError(msg)


def random_split_like_tree(rng_key, target=None, treedef=None):
    # https://github.com/google/jax/discussions/9508
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target, n_samples: Optional[int] = None):
    # https://github.com/google/jax/discussions/9508
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda ell, k: jax.random.normal(k, ell.shape, ell.dtype), target, keys_tree
        )
    return jax.tree_util.tree_map(
        lambda ell, k: jax.random.normal(k, (n_samples, *ell.shape), ell.dtype),
        target,
        keys_tree,
    )


@partial(jax.jit, static_argnames=("computation_type", "n_samples"))
def hutchinson_diagonal(
    gvp_fn,
    params,
    n_samples,
    key,
    num_levels=5,
    computation_type: Literal["serial", "parallel"] = "serial",
):
    """ "
    This function computes the diagonal of the GGN matrix using Hutchinson's method.
    """

    diag_init = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    if computation_type == "serial":

        def diag_estimate_fn(key, control_variate):
            diag_init_ = jax.tree_map(lambda x: jnp.zeros_like(x), params)
            key_list = jax.random.split(key, n_samples)

            def single_eps_diag(n, res):
                diag = res
                key = key_list[n]
                eps = tree_random_normal_like(key, control_variate)
                c_v = tm.Vector(control_variate) * tm.Vector(eps)
                gvp = gvp_fn(jax.flatten_util.ravel_pytree(eps)[0])
                new_diag = (
                    tm.Vector(eps) * (tm.Vector(gvp) - c_v) + tm.Vector(control_variate)
                ).tree
                return jax.tree_map(lambda x, y: x + y, new_diag, diag)

            diag = jax.lax.fori_loop(0, n_samples, single_eps_diag, diag_init_)
            return jax.tree_map(lambda x: x / n_samples, diag)
    elif computation_type == "parallel":

        def diag_estimate_fn(key, control_variate):
            key_list = jax.random.split(key, n_samples)

            @jax.vmap
            def single_eps_diag(key):
                eps = tree_random_normal_like(key, control_variate)
                c_v = tm.Vector(control_variate) * tm.Vector(eps)
                gvp = gvp_fn(jax.flatten_util.ravel_pytree(eps)[0])
                return (
                    tm.Vector(eps) * (tm.Vector(gvp) - c_v) + tm.Vector(control_variate)
                ).tree

            diag = single_eps_diag(key_list)
            return jax.tree_map(lambda x: x.mean(axis=0), diag)

    def body_fun(n, res):
        diag, key = res
        key, subkey = jax.random.split(key)
        diag_update = diag_estimate_fn(subkey, diag)
        diag = ((tm.Vector(diag) * n + tm.Vector(diag_update)) / (n + 1)).tree
        return (diag, key)

    diag, _ = jax.lax.fori_loop(0, num_levels, body_fun, (diag_init, key))
    return diag


@partial(jax.jit, static_argnames=("likelihood", "model_fn"))
def last_layer_ggn(
    model_fn,
    params,
    x_batch,
    likelihood: Literal["classification", "regression"] = "classification",
):
    leafs, _ = jax.tree_util.tree_flatten(params)
    N_llla = len(leafs[-1]) + len(leafs[-2])
    params_vec, unflatten_fn = jax.flatten_util.ravel_pytree(params)

    def model_apply_vec(params_vectorized, x):
        return model_fn(unflatten_fn(params_vectorized), x)

    def last_layer_model_fn(last_params_vec, first_params, x):
        first_params = jax.lax.stop_gradient(first_params)
        vectorized_params = jnp.concatenate([first_params, last_params_vec])
        return model_apply_vec(vectorized_params, x)

    params_ll = params_vec[-N_llla:]
    J_ll = jax.jacfwd(last_layer_model_fn, argnums=0)(
        params_ll, params_vec[:-N_llla], x_batch
    )
    # B , O , N_llla
    if likelihood == "regression":
        J_ll = J_ll.reshape(-1, N_llla)
        return J_ll.T @ J_ll

    if likelihood == "classification":
        pred = model_fn(params, x_batch)  # B, O
        pred = jax.nn.softmax(pred, axis=1)
        pred = jax.lax.stop_gradient(pred)
        D = jax.vmap(jnp.diag)(pred)
        H = jnp.einsum("bo, bi->boi", pred, pred)
        H = D - H  # B, O, O
        return jnp.einsum("mob, boo, bon->mn", J_ll.T, H, J_ll)
    raise ValueError
