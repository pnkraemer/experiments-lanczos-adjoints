"""
Script for training/calibrating Gaussian process models on UCI datasets.

Currently, use either of the following datasets:
* concrete_compressive_strength  (small) ~ 1.000
* combined_cycle_power_plant  (medium) ~ 10.000
* ___________________________   (large)
* ___________________________   (very large)
"""

import argparse
import json
import os
import time
from typing import Literal, get_args

import gpytorch
import jax
import jax.numpy as jnp
import optax
import torch
import tqdm
from matfree import hutchinson
from matfree_extensions.util import data_util, exp_util
from matfree_extensions.util import gp_util as gp


def save_parser(directory, args):
    if args.gp_method == "adjoints":
        with open(
            f"{directory}/args_config_"
            + args.gp_method
            + f"_kr{args.slq_krylov_depth!s}.json",
            "w",
        ) as f:
            json.dump(vars(args), f, indent=4)
    else:
        with open(f"{directory}/args_config_" + args.gp_method + ".json", "w") as f:
            json.dump(vars(args), f, indent=4)


def load_parser(directory, args, parser):
    with open(f"{directory}/args_config_" + args.gp_method + ".json") as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)


# 1,000     ---> concrete_compressive_strength
# 10,000    ---> combined_cycle_power_plant
# 200,000   ---> sgemmgpu
# 2,000,000 ---> household electric
# 4,000,000 ---> gassensors

GP_METHODS_ARGS = Literal["naive", "gpytorch", "adjoints", "gpjax", "cg"]
GP_METHODS = get_args(GP_METHODS_ARGS)


# Base GPyTorch GP Regression model for their solvers.
class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_):
        super().__init__(train_x, train_y, likelihood_)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def data_train_test_split_80_20(X, y, /, *, key):
    """Split a data set into a training and a testing part."""
    ints = jnp.arange(0, len(X))
    ints_shuffled = jax.random.permutation(key, ints)

    idx = len(X) // 5  # 20 percent test data
    test, train = ints_shuffled[:idx], ints_shuffled[idx:]
    return (X[train], y[train]), (X[test], y[test])


def init_params_as_gpytorch(X, y):
    """Returns the same initialization hyperparameters as in GPyTorch"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = _ExactGPModel(X[0, :], y[0], likelihood)

    lengthscale = jnp.asarray(
        model.covar_module.base_kernel.raw_lengthscale.detach().numpy()
    )
    outputscale = jnp.asarray(model.covar_module.raw_outputscale.detach().numpy())
    noise = jnp.asarray(likelihood.raw_noise.detach().numpy())

    return ((lengthscale, outputscale), noise)


def gp_select(which: GP_METHODS_ARGS, X, y, key, solver_args):
    """Selects a GP given one methods we are interested in"""

    # todo: ideally, additionally to the reference, we should pass
    # the solver of the other method we want to combine to compute
    # the MLL during training

    if which not in GP_METHODS:
        msg = "The dataset is unknown."
        msg += f"\n\tExpected: One of {GP_METHODS}."
        msg += f"\n\tReceived: '{which}'."
        raise ValueError(msg)

    if which == "naive" or which == "adjoints":
        _X = jnp.asarray(X.detach().cpu().numpy())
        _y = jnp.asarray(y.detach().cpu().numpy()).ravel()

        # Log-pdf function
        if which == "naive":
            logpdf_fun = gp.logpdf_cholesky()
        elif which == "adjoints":
            num_batches = solver_args["num_batches"]
            num_samples = solver_args["num_samples"]
            krylov_depth = solver_args["krylov_depth"]
            # num_batches ---> 1 and larger samples as possible.
            # begin to split batches and lower samples if memory issues
            # krylov_depth upper bounded by num. of points
            # (roughly the num of eigens)

            x_like = jnp.ones((_X.shape[0],), dtype=float)
            sampler = hutchinson.sampler_rademacher(
                x_like, num=num_samples
            )  # make an option for normal?
            logpdf_fun = gp.logpdf_lanczos(
                krylov_depth, sampler, slq_batch_num=num_batches
            )

        # THIS GUY FOR SMALL DATA REGIME
        # gram_matvec = (
        #     gp.gram_matvec_full_batch()
        # )  # I chose this one, but any other could be possible
        # THIS GUY FOR SMALL DATA REGIME
        print("HERE")
        gram_matvec = (
            gp.gram_matvec_map_over_batch(batch_size=2)
            # gp.gram_matvec_map()
            # gp.gram_matvec_full_batch()
        )

        # Set up a GP model
        shape_in = jnp.shape(_X[0])
        shape_out = jnp.shape(_y[0])
        k, p_prior = gp.kernel_scaled_rbf(shape_in=shape_in, shape_out=shape_out)
        prior = gp.model(gp.mean_zero(), k, gram_matvec=gram_matvec)
        likelihood, p_likelihood = gp.likelihood_gaussian()
        loss = gp.mll_exact(prior, likelihood, logpdf=logpdf_fun)

        # Ensure that the parameters match
        ((lengthscale, outputscale), noise) = init_params_as_gpytorch(X, y)
        p_prior["raw_lengthscale"] = lengthscale.squeeze()
        p_prior["raw_outputscale"] = outputscale.squeeze()
        p_likelihood["raw_noise"] = noise.squeeze()

        reference = (_X, _y), loss, ((p_prior, p_likelihood), key)

    elif which == "gpytorch":
        # output_device = torch.device('cuda:0')
        # X, y = X.to(output_device), y.to(output_device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        model = _ExactGPModel(X, y, likelihood).cuda()

        # sets gpytorch classes in training mode (not eval)
        model.train()
        likelihood.train()

        loss = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        reference = (X, y), loss, model

    elif which == "gpjax" or which == "cg":
        pass

    return reference


def gp_train(which: GP_METHODS_ARGS, reference, num_epochs):
    """Performs the specific training loop for a given GP method"""

    if which not in GP_METHODS:
        msg = "The dataset is unknown."
        msg += f"\n\tExpected: One of {GP_METHODS}."
        msg += f"\n\tReceived: '{which}'."
        raise ValueError(msg)

    # convergence curves + time-stamps
    loss_curve = []
    loss_timestamps = []

    if which == "naive" or which == "adjoints" or which == "gpjax" or which == "cg":
        # getting data, loss and model/params
        (X, y), loss, ((p_prior, p_likelihood), key) = reference

        p_opt, unflatten = jax.flatten_util.ravel_pytree([p_prior, p_likelihood])
        optimizer = optax.adam(learning_rate=0.1)
        state = optimizer.init(p_opt)

        def mll(params, *params_logpdf, inputs, targets):
            p1, p2 = unflatten(params)
            return -loss(
                inputs, targets, *params_logpdf, params_prior=p1, params_likelihood=p2
            )

        # for LANCZOS we need to specify the key
        def mll_lanczos(params, key, inputs, targets):
            p1, p2 = unflatten(params)
            return -loss(inputs, targets, key, params_prior=p1, params_likelihood=p2)

        if which == "naive":
            value_and_grad_gp = jax.jit(jax.value_and_grad(mll, argnums=0))
            value, _grad = value_and_grad_gp(p_opt, inputs=X, targets=y)
        elif which == "adjoints":
            value_and_grad_gp = jax.jit(jax.value_and_grad(mll_lanczos, argnums=0))
            value, _grad = value_and_grad_gp(p_opt, key, inputs=X, targets=y)

        progressbar = tqdm.tqdm(range(args.num_epochs))
        progressbar.set_description(f"loss: {value:.3F}")
        start = time.perf_counter()

        loss_curve.append(float(value))
        loss_timestamps.append(start - start)

        for _ in progressbar:
            try:
                # todo: if we use lanjczos, split the random key HERE
                if which == "naive":
                    value, grads = value_and_grad_gp(p_opt, inputs=X, targets=y)
                elif which == "adjoints":
                    key, subkey = jax.random.split(key, num=2)
                    value, grads = value_and_grad_gp(
                        p_opt, subkey, inputs=X, targets=y
                    )

                updates, state = optimizer.update(grads, state)
                p_opt = optax.apply_updates(p_opt, updates)
                progressbar.set_description(f"loss: {value:.3F}")

                current = time.perf_counter()
                loss_curve.append(float(value))
                loss_timestamps.append(current - start)

            except KeyboardInterrupt:
                break
        end = time.perf_counter()
        opt_params = p_opt

    elif which == "gpytorch":
        # getting data, loss and model/params
        (X, y), loss, model = reference

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        progressbar = tqdm.tqdm(range(num_epochs))
        start = time.perf_counter()

        value = -loss(model(X), y)
        loss_curve.append(value.item())
        loss_timestamps.append(start - start)

        for _ in progressbar:
            try:
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Calculate loss and backprop gradients
                value = -loss(model(X), y)
                value.backward()
                optimizer.step()
                progressbar.set_description(f"loss: {value.item():.3F}")

                current = time.perf_counter()
                loss_curve.append(value.item())
                loss_timestamps.append(current - start)

            except KeyboardInterrupt:
                break
        end = time.perf_counter()

        lengthscale = (
            model.covar_module.base_kernel.raw_lengthscale.detach().cpu().numpy()
        )
        outputscale = model.covar_module.raw_outputscale.detach().cpu().numpy()
        noise = model.likelihood.raw_noise.detach().cpu().numpy()

        opt_params = [lengthscale.item(), outputscale.item(), noise.item()]

    return end - start, (loss_curve, loss_timestamps), opt_params  # save stuff here


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_of_file", "-nof", type=str, default="gp_train")
    parser.add_argument("--seed", "-seed", type=int, default=1)
    parser.add_argument("--num_points", "-n", type=int, default=-1)
    parser.add_argument("--num_epochs", "-e", type=int, default=100)
    parser.add_argument("--gp_method", "-gpm", type=str, default="gpytorch")
    parser.add_argument("--slq_num_batches", "-sb", type=int, default=100)
    parser.add_argument("--slq_krylov_depth", "-kry", type=int, default=5)
    parser.add_argument("--slq_num_samples", "-s", type=int, default=1)  # higher?
    parser.add_argument(
        "--dataset", "-data", type=str, default="concrete_compressive_strength"
    )
    args = parser.parse_args()
    print(args, "\n")

    # Initialise the random number generator
    key_ = jax.random.PRNGKey(args.seed)
    key_data, key_init, key_solver = jax.random.split(key_, num=3)

    # Load the data
    (X_train, y_train), (X_test, y_test) = data_util.load_uci_data(
        args.dataset, args.seed
    )

    # Select GP method with solver args dict (for lanczos)
    args_solver = {
        "num_batches": args.slq_num_batches,
        "krylov_depth": args.slq_krylov_depth,
        "num_samples": args.slq_num_samples,
    }
    reference = gp_select(args.gp_method, X_train, y_train, key_solver, args_solver)

    # Training the GP with different methods/matrix-solvers
    start = time.perf_counter()
    train_time, (conv, tstamp), params = gp_train(
        args.gp_method, reference, args.num_epochs
    )

    print(f"(epochs={args.num_epochs:.0f},", f"training time={train_time:.3f})")

    # Create a directory for the results
    dir_local = exp_util.matching_directory(__file__, "results/")
    os.makedirs(dir_local, exist_ok=True)

    # Saving {convergence_curve, convergence time_stamps, optimized parameters}
    if args.gp_method == "adjoints":
        jnp.save(
            f"{dir_local}/convergence_{args.gp_method}_kr{args.slq_krylov_depth!s}.npy",
            jnp.array(conv),
        )
        jnp.save(
            f"{dir_local}/time_{args.gp_method}" + f"_kr{args.slq_krylov_depth!s}.npy",
            jnp.array(tstamp),
        )
        jnp.save(
            f"{dir_local}/params_{args.gp_method}"
            + f"_kr{args.slq_krylov_depth!s}.npy",
            jnp.array(params),
        )
    else:
        jnp.save(f"{dir_local}/convergence_{args.gp_method}.npy", jnp.array(conv))
        jnp.save(f"{dir_local}/time_{args.gp_method}.npy", jnp.array(tstamp))
        jnp.save(f"{dir_local}/params_{args.gp_method}.npy", jnp.array(params))

    # Saving {argparse configuration}
    save_parser(dir_local, args)
