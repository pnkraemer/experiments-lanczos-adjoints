import argparse
import os
import os.path
import time

import gpytorch
import gpytorch.settings as cfg
import jax.numpy as jnp
import numpy as onp  # noqa: ICN001
import torch
import tqdm
from matfree_extensions.util import exp_util, uci_util


def load_data(which: str, /):
    if which == "concrete":
        return uci_util.uci_concrete()
    if which == "power_plant":
        return uci_util.uci_power_plant()
    if which == "parkinson":
        return uci_util.uci_parkinson()
    if which == "protein":
        return uci_util.uci_protein()
    if which == "bike_sharing":
        return uci_util.uci_bike_sharing()
    if which == "kegg_undirected":
        return uci_util.uci_kegg_undirected()
    if which == "kegg_directed":
        return uci_util.uci_kegg_directed()
    if which == "elevators":
        return uci_util.uci_elevators()
    if which == "kin40k":
        return uci_util.uci_kin40k()
    if which == "slice":
        return uci_util.uci_slice()
    raise ValueError


# Choose parameters
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--num_partitions", type=int, required=True)
parser.add_argument("--rank_precon", type=int, required=True)
parser.add_argument("--num_matvecs", type=int, required=True)
parser.add_argument("--num_samples", type=int, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--cg_tol", type=float, required=True)
args = parser.parse_args()
print(args)


# Process parameters
torch.manual_seed(args.seed)
num_matvecs_train_lanczos = args.num_matvecs

# Load data
inputs, targets = load_data(args.dataset)
inputs = torch.from_numpy(onp.asarray(inputs).copy())
targets = torch.from_numpy(onp.asarray(targets).copy())
idx = torch.randperm(len(inputs))
inputs, targets = inputs[idx], targets[idx]

# Subsample data to a multiple of num_partitions
# This script doesn't have to, but we the other code does.
num_data_raw = len(inputs)
coeff = num_data_raw // (5 * args.num_partitions)
num_data = int(coeff * 5 * args.num_partitions)
# print(
#     f"Subsampling data from N={num_data_raw} points "
#     f"to N={num_data} points to match "
#     f"P={args.num_partitions} partitions "
#     f"and the train-test-split "
#     f"from the other code."
# )
inputs, targets = inputs[:num_data], targets[:num_data]

# Configs
with (
    cfg.max_preconditioner_size(args.rank_precon),
    cfg.cg_tolerance(args.cg_tol),
    cfg.num_trace_samples(args.num_samples),
    cfg.max_lanczos_quadrature_iterations(args.num_matvecs),
    cfg.ciq_samples(False),
    cfg.deterministic_probes(False),
    cfg.skip_logdet_forward(False),
    cfg.fast_computations(True, True, True),
    cfg.max_root_decomposition_size(args.num_matvecs),
    cfg.min_preconditioning_size(10),
    cfg.tridiagonal_jitter(0.0),
):
    # Make train/test split
    n_train = int(0.8 * len(inputs))
    train_x, train_y = inputs[:n_train], targets[:n_train]
    test_x, test_y = inputs[n_train:], targets[n_train:]

    # Make contiguous
    train_x, train_y = train_x.contiguous(), train_y.contiguous()
    test_x, test_y = test_x.contiguous(), test_y.contiguous()

    # Put on CUDA
    output_device = torch.device("cuda:0")
    train_x, train_y = train_x.to(output_device), train_y.to(output_device)
    test_x, test_y = test_x.to(output_device), test_y.to(output_device)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x_, train_y_, likelihood_):
            super().__init__(train_x_, train_y_, likelihood_)
            ndims = train_x_.size(-1)
            kernel = gpytorch.kernels.keops.MaternKernel(nu=1.5, ard_num_dims=ndims)
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
            self.mean_module = gpytorch.means.ConstantMean()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Initialise likelihood and model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # model = ExactGPModel(train_x, train_y, likelihood)
    positive = gpytorch.constraints.GreaterThan(1e-4)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=positive
    ).cuda()
    model = ExactGPModel(train_x, train_y, likelihood).cuda()

    # Initialise randomly
    hypers = {
        "likelihood.noise_covar.raw_noise": torch.randn(()).cuda(),
        "covar_module.base_kernel.raw_lengthscale": torch.randn(
            (train_x.size(-1),)
        ).cuda(),
        "covar_module.raw_outputscale": torch.randn(()).cuda(),
        "mean_module.raw_constant": torch.randn(()).cuda(),
    }
    model.initialize(**hypers)

    # Train with Adam
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # optimizer = torch.optim.LBFGS(model.parameters())
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Prepare saving some results
    gradient_norms: dict = {}
    for name, _p in model.named_parameters():
        gradient_norms[name] = []
    loss_values = []
    loss_timestamps = []

    output = model(train_x)
    loss = -mll(output, train_y)

    progressbar = tqdm.tqdm(range(args.num_epochs))
    progressbar.set_description(f"Loss {loss:.9f}")
    time_start = time.perf_counter()

    for _ in progressbar:
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            # Store values
            loss_values.append(loss)
            loss_timestamps.append(time.perf_counter() - time_start)

            progressbar.set_description(f"Loss {loss:.3f}")
        except KeyboardInterrupt:
            break


# Evaluate:
model.eval()
likelihood.eval()
# Configs
with (
    torch.no_grad(),
    cfg.max_preconditioner_size(args.rank_precon),
    cfg.cg_tolerance(1e-4),
    cfg.num_trace_samples(args.num_samples),
    cfg.max_lanczos_quadrature_iterations(args.num_matvecs),
    cfg.ciq_samples(False),
    cfg.deterministic_probes(False),
    cfg.skip_logdet_forward(False),
    cfg.fast_computations(True, True, True),
    cfg.max_root_decomposition_size(args.num_matvecs),
    cfg.min_preconditioning_size(10),
    cfg.tridiagonal_jitter(0.0),
):
    with cfg.skip_posterior_variances():
        pred_dist = likelihood(model(test_x))
        mean = pred_dist.mean
        rmse = mean.sub(test_y).pow(2).mean().sqrt()

    # testloss = -mll(model(test_x), test_y)
    # print("Test-loss:", testloss)
    print("RMSE:", rmse)


# Save results to a file
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)
path = f"{directory}{args.name}_{args.dataset}_s{args.seed}"

for name, _ in model.named_parameters():
    array = jnp.asarray(torch.tensor(gradient_norms[name]).numpy())
    jnp.save(f"{path}_gradient_norms_{name}.npy", array)

array = jnp.asarray(torch.tensor(loss_values).numpy())
jnp.save(f"{path}_loss_values.npy", array)

array = jnp.asarray(torch.tensor(loss_timestamps).numpy())
jnp.save(f"{path}_loss_timestamps.npy", array)

array = jnp.asarray(rmse.detach().cpu().numpy())
jnp.save(f"{path}_rmse.npy", array)
# array = jnp.asarray(testloss.detach().cpu().numpy())
# jnp.save(f"{path}_testloss.npy", array)

# print()
print()
print()
