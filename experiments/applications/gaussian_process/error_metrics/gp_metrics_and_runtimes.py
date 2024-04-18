"""RMSE, NLL and Runtime of Gaussian process models on a UCI dataset.

Currently, use either of the following datasets:
* uci_concrete_compressive_strength  (small)
* uci_combined_cycle_power_plant  (medium)
* ___________________________   (large)
* ___________________________   (very large)
"""

import argparse
import os
import time
from typing import Literal, get_args

import gpytorch
import jax
import jax.numpy as jnp
from matfree_extensions import data_util, exp_util

# 1,000     ---> concrete_compressive_strength
# 10,000    ---> combined_cycle_power_plant
# 200,000   ---> sgemmgpu
# 2,000,000 ---> household electric
# 4,000,000 ---> gassensors

GP_METHODS_ARGS = Literal["naive", "gpytorch", "adjoints", "gpjax", "cg"]
GP_METHODS = get_args(GP_METHODS_ARGS)


def data_train_test_split_80_20(X, y, /, *, key):
    """Split a data set into a training and a testing part."""
    ints = jnp.arange(0, len(X))
    ints_shuffled = jax.random.permutation(key, ints)

    idx = len(X) // 5  # 20 percent test data
    test, train = ints_shuffled[:idx], ints_shuffled[idx:]
    return (X[train], y[train]), (X[test], y[test])


def gp_method_select(which: GP_METHODS_ARGS, X, y):
    """Selects a GP given one methods we are interested in"""

    # todo: ideally, additionally to the reference, we should pass
    # the solver of the other method we want to combine to compute
    # the MLL during training

    if which not in GP_METHODS:
        msg = "The dataset is unknown."
        msg += f"\n\tExpected: One of {GP_METHODS}."
        msg += f"\n\tReceived: '{which}'."
        raise ValueError(msg)

    if which == "naive":
        pass

    elif which == "gpytorch":

        class _ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood_):
                super().__init__(train_x, train_y, likelihood_)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = _ExactGPModel(X, y, likelihood)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        value = mll(model(X), y)

        lengthscale = model.covar_module.base_kernel.raw_lengthscale.item()
        outputscale = model.covar_module.raw_outputscale.item()
        noise = likelihood.raw_noise.item()
        params = ((lengthscale, outputscale), noise)

        reference = (X, y), value, params

    elif which == "adjoints" or which == "gpjax" or which == "cg":
        pass

    return reference


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_of_run", type=str, default="")
    parser.add_argument("--seed", "-s", type=int, default=1)
    parser.add_argument("--num_points", "-n", type=int, default=-1)
    parser.add_argument("--num_epochs", "-e", type=int, default=10)
    parser.add_argument("--gp_method", "-gpm", type=str, default="gpytorch")
    parser.add_argument(
        "--dataset", "-data", type=str, default="concrete_compressive_strength"
    )
    args = parser.parse_args()
    print(args, "\n")

    # Initialise the random number generator
    key_ = jax.random.PRNGKey(args.seed)
    key_data, key_init, key_solve = jax.random.split(key_, num=3)

    # Load the data
    (X_train, y_train), (X_test, y_test) = data_util.load_uci_data(
        args.dataset, args.seed
    )

    # Select GP method
    reference = gp_method_select(args.gp_method, X_train, y_train)

    # Training the GP with different methods/matrix-solvers
    start = time.time()
    end = time.time()

    print(
        # "(rmse={:.3f},".format(rmse),
        # "nll={:.3f},".format(nll),
        f"epochs/iterations={args.num_epochs:.0f},",
        f"training time={end - start:.0f})",
    )

    # Create a directory for the results
    directory_local = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory_local, exist_ok=True)
