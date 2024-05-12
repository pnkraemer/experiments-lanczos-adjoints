import os.path
import time
import urllib.request

import gpytorch
import torch
from scipy.io import loadmat

if not os.path.isfile("../3droad.mat"):
    print("Downloading '3droad' UCI dataset...")
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1", "../3droad.mat"
    )

data = torch.Tensor(loadmat("../3droad.mat")["data"])


# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

device = "cuda:0"

# N = data.shape[0]
N = 40_000
training_iter = 100
num_samples = 10
num_matvecs_train_lanczos = 10
num_matvecs_train_cg = 100 * num_matvecs_train_lanczos
num_matvecs_eval_cg = 100 * num_matvecs_train_cg
rank_precon = 500

# make train/val/test
n_train = int(0.9 * N)
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train:N, :-1], data[n_train:N, -1]

# normalize features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6  # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# normalize labels
mean, std = train_y.mean(), train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

output_device = torch.device("cuda:0")

train_x, train_y = train_x.to(output_device), train_y.to(output_device)
test_x, test_y = test_x.to(output_device), test_y.to(output_device)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
model = ExactGPModel(train_x, train_y, likelihood).cuda()

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


# with (gpytorch.settings.max_preconditioner_size(500),gpytorch.settings.cg_tolerance(1e0), gpytorch.settings.max_cg_iterations(100_000), gpytorch.settings.verbose_linalg(True)):
# with gpytorch.settings.verbose_linalg(True):
# with gpytorch.settings.verbose_linalg(True), gpytorch.settings.cg_tolerance(1e0), gpytorch.settings.max_preconditioner_size(250), gpytorch.settings.max_cg_iterations(100_000), gpytorch.settings.linalg_dtypes(default=torch.float32):
with gpytorch.settings.max_preconditioner_size(
    rank_precon
), gpytorch.settings.cg_tolerance(1e-2), gpytorch.settings.num_trace_samples(
    num_samples
), gpytorch.settings.max_lanczos_quadrature_iterations(
    num_matvecs_train_lanczos
), gpytorch.settings.deterministic_probes(True), gpytorch.settings.max_cg_iterations(
    num_matvecs_train_cg
), gpytorch.settings.verbose_linalg(False):
    for i in range(training_iter):
        try:
            start_time = time.time()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()

            for name, p in model.named_parameters():
                print(name, p.grad)

            # print("CG iterations {:4d}".format(len(gpytorch.settings.record_residual.lst_residual_norm)))
            print(
                "Iter %d/%d - Loss: %.3f    noise: %.3f"
                % (i + 1, training_iter, loss.item(), model.likelihood.raw_noise.item())
            )
            optimizer.step()
            # print(time.time() - start_time)
            print()
        except KeyboardInterrupt:
            break

    # RMSE
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.skip_posterior_variances():
    pred_dist = likelihood(model(test_x))
    mean = pred_dist.mean
    rmse = mean.sub(test_y).pow(2).mean().sqrt()
    print("RMSE", rmse)
