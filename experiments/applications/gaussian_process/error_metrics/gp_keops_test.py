import argparse
import os
import os.path
import urllib.request

import gpytorch
import gpytorch.settings as cfg
import jax.numpy as jnp
import torch
import tqdm
from matfree_extensions.util import exp_util
from scipy.io import loadmat

# Load the dataset
if not os.path.isfile("../3droad.mat"):
    print("Downloading '3droad' UCI dataset...")
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1", "../3droad.mat"
    )

data = torch.Tensor(loadmat("../3droad.mat")["data"])


# Choose parameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_data", type=int, required=True)
parser.add_argument("--rank_precon", type=int, required=True)
parser.add_argument("--num_matvecs", type=int, required=True)
parser.add_argument("--num_samples", type=int, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
args = parser.parse_args()
print(args)

# Process parameters
torch.manual_seed(args.seed)
num_matvecs_train_lanczos = args.num_matvecs
num_matvecs_train_cg = 2 * num_matvecs_train_lanczos
num_matvecs_eval_cg = 100 * num_matvecs_train_cg

# Make train/test split
n_train = int(0.9 * args.num_data)
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train : args.num_data, :-1], data[n_train : args.num_data, -1]

# Normalise features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6  # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# Normalise labels
mean, std = train_y.mean(), train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std

# Make contiguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

# Put on CUDA
# output_device = torch.device("cuda:0")
# train_x, train_y = train_x.to(output_device), train_y.to(output_device)
# test_x, test_y = test_x.to(output_device), test_y.to(output_device)


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
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
# likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
# model = ExactGPModel(train_x, train_y, likelihood).cuda()


# Train with Adam
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
progressbar = tqdm.tqdm(range(args.num_epochs))
progressbar.set_description(f"Loss {1000.:.3f}, noise: {1000.:.3f}")

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


# Prepare saving some results
gradient_norms: dict = {}
for name, _p in model.named_parameters():
    gradient_norms[name] = []
loss_values = []

# Configs
cfg_precon = cfg.max_preconditioner_size(args.rank_precon)
cfg_cg_tol = cfg.cg_tolerance(1e-2)
cfg_smpls = cfg.num_trace_samples(args.num_samples)
cfg_lanczos = cfg.max_lanczos_quadrature_iterations(num_matvecs_train_lanczos)
cfg_probes = cfg.deterministic_probes(True)
cfg_cg_maxiter = cfg.max_cg_iterations(num_matvecs_train_cg)
with cfg_precon, cfg_cg_tol, cfg_smpls, cfg_lanczos, cfg_probes, cfg_cg_maxiter:
    # Start training
    for _ in progressbar:
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()

            # Store values
            loss_values.append(loss)
            for name, p in model.named_parameters():
                gradient_norms[name].append(torch.norm(p.grad))

            raw_noise = model.likelihood.raw_noise.item()
            progressbar.set_description(f"Loss {loss:.3f}, noise: {raw_noise:.3f}")

            optimizer.step()
        except KeyboardInterrupt:
            break


# Evaluate:
model.eval()
likelihood.eval()

with torch.no_grad():
    # RMSE
    cfg_cg_maxiter = cfg.max_cg_iterations(num_matvecs_eval_cg)
    cfg_cg_tol = cfg.eval_cg_tolerance(1e-4)
    cfg_skip_var = cfg.skip_posterior_variances()
    cfg_comp = cfg.fast_computations(False, False, False)
    with cfg_cg_maxiter, cfg_cg_tol, cfg_skip_var, cfg_comp:
        pred_dist = likelihood(model(test_x))
        mean = pred_dist.mean
        rmse = mean.sub(test_y).pow(2).mean().sqrt()

    # NLL ??

    # Save results to a file
    directory = exp_util.matching_directory(__file__, "results/")
    os.makedirs(directory, exist_ok=True)

    for name, _ in model.named_parameters():
        array = jnp.asarray(torch.tensor(gradient_norms[name]).numpy())
        jnp.save(f"{directory}gpytorch_gradient_norms_{name}.npy", array)

    array = jnp.asarray(torch.tensor(loss_values).numpy())
    jnp.save(f"{directory}gpytorch_loss_values.npy", array)

    array = jnp.asarray(rmse.numpy())
    jnp.save(f"{directory}gpytorch_rmse.npy", array)

    print()
