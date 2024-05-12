import argparse
import os.path
import time
import urllib.request

import jax
import jax.numpy as jnp
import jaxopt
import scipy.io
import tqdm
from matfree import hutchinson
from matfree_extensions import low_rank
from matfree_extensions.util import data_util, exp_util, gp_util, gp_util_linalg


def root_mean_square_error(x, *, target):
    error_abs = x - target
    return jnp.linalg.norm(error_abs) / jnp.sqrt(x.size)


if not os.path.isfile("../3droad.mat"):
    print("Downloading '3droad' UCI dataset...")
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1", "../3droad.mat"
    )

data = jnp.asarray(scipy.io.loadmat("../3droad.mat")["data"])
print(jnp.shape(data))

# Choose parameters
# todo: give cg fewer partitions than slq, because cg does not track batched samples!
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1)

args = parser.parse_args()
print(args)


num_data = 40
num_matvecs_train_lanczos = 10
num_matvecs_train_cg = 50  # match num_samples * num_matvecs_lanczos?
num_matvecs_eval_cg = 20
num_samples_batched = 1
num_samples_sequential = 5
num_partitions_train = 1
rank_precon = 5
num_epochs = 50

memory_bytes = (
    num_data**2
    * num_matvecs_train_lanczos
    * num_samples_batched
    / num_partitions_train
    * 32
)
memory_gb = memory_bytes / 8589934592
print(f"\nPredicting ~ {memory_gb} GB of memory\n")

key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key, num=2)
data_sampled = data[:num_data, :-1], data[:num_data, -1]
train, test = data_util.split_train_test_shuffle(subkey, *data_sampled, train=0.9)
# train, test = data_util.split_train_test(*data_sampled, train=0.9)
(train_x, train_y), (test_x, test_y) = train, test
print("Train:", train_x.shape, train_y.shape)
print("Test:", test_x.shape, test_y.shape)


# Normalise features
mean = train_x.mean(axis=-2, keepdims=True)
std = train_x.std(axis=-2, keepdims=True) + 1e-6  # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# Normalise labels
mean, std = train_y.mean(), train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std


# Set up linear algebra for training
solve_p = gp_util_linalg.krylov_solve_pcg_fixed_step(num_matvecs_train_cg)
v_like = jnp.ones((len(train_x),), dtype=float)
sample = hutchinson.sampler_rademacher(v_like, num=num_samples_batched)
logdet = gp_util_linalg.krylov_logdet_slq(
    num_matvecs_train_lanczos, sample=sample, num_batches=num_samples_sequential
)
cholesky = low_rank.cholesky_partial_pivot(rank=rank_precon)
precondition = low_rank.preconditioner(cholesky)
logpdf_p = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)
gram_matvec = gp_util_linalg.gram_matvec_map_over_batch(
    num_batches=num_partitions_train
)
likelihood, p_likelihood = gp_util.likelihood_gaussian_pdf_p(
    gram_matvec, logpdf_p, precondition
)

# Set up a model
m, p_mean = gp_util.mean_constant(shape_out=())
k, p_kernel = gp_util.kernel_scaled_matern_32(shape_in=(3,), shape_out=())
prior = gp_util.model(m, k)

# Build the loss and evaluate
loss = gp_util.mll_exact(prior, likelihood)

# Set up matrix-free linear algebra
# todo: why does solve_pcg_fixed_step_reortho nan out??

# Initialise the parameters
key, subkey = jax.random.split(key)
ps = exp_util.tree_random_like(subkey, (p_mean, p_kernel, p_likelihood))
p_opt, unflatten = jax.flatten_util.ravel_pytree(ps)


# Evaluate the loss function
@jax.jit
def mll_lanczos(params, *p_logdet, inputs, targets):
    p1, p2, p3 = unflatten(params)
    val, info = loss(
        inputs,
        targets,
        *p_logdet,
        params_mean=p1,
        params_kernel=p2,
        params_likelihood=p3,
    )
    return -val, info


@jax.jit
def mll_cholesky(params, inputs, targets):
    p1, p2, p3 = unflatten(params)

    # Build the loss and evaluate
    logpdf = gp_util.logpdf_scipy_stats()
    lklhd, _ = gp_util.likelihood_gaussian_pdf(gram_matvec, logpdf)

    loss_fun = gp_util.mll_exact(prior, lklhd)
    val, info = loss_fun(
        inputs, targets, params_mean=p1, params_kernel=p2, params_likelihood=p3
    )
    return -val, info


# Use a Krylov solver with 2x as many steps for evaluation
solve = gp_util_linalg.krylov_solve_cg_fixed_step(num_matvecs_eval_cg)
likelihood_, _p_likelihood_ = gp_util.likelihood_gaussian_condition(gram_matvec, solve)

posterior = gp_util.posterior_exact(prior, likelihood_)


@jax.jit
def predict_mean(params, x, inputs, targets):
    p1, p2, p3 = unflatten(params)

    postmean, _ = posterior(
        inputs=inputs,
        targets=targets,
        params_mean=p1,
        params_kernel=p2,
        params_likelihood=p3,
    )
    return postmean(x)


# Pre-compile the loss function
key, subkey = jax.random.split(key)
(mll_train, aux) = mll_lanczos(p_opt, subkey, inputs=train_x, targets=train_y)
mll_train.block_until_ready()
print(aux)
residual = aux["logpdf"]["residual"]
cg_error = jnp.linalg.norm(residual) / jnp.sqrt(len(residual))

# Benchmark the loss function
t0 = time.perf_counter()
for _ in range(1):
    (value, aux) = mll_lanczos(p_opt, subkey, inputs=train_x, targets=train_y)
    value.block_until_ready()
t1 = time.perf_counter()
print("Runtime (value):", (t1 - t0) / 1)


# Pre-compile the value-and-grad
value_and_grad = jax.jit(jax.value_and_grad(mll_lanczos, argnums=0, has_aux=True))
_, grad_train = value_and_grad(p_opt, subkey, inputs=train_x, targets=train_y)
grad_train.block_until_ready()


# Benchmark the value-and-gradient function
t0 = time.perf_counter()
for _ in range(1):
    (value, _aux), grad = value_and_grad(p_opt, key, inputs=train_x, targets=train_y)
    value.block_until_ready()
    grad.block_until_ready()
t1 = time.perf_counter()
print("Runtime (value-and-gradient):", (t1 - t0) / 1)


# Pre-compile the test-loss
predicted, _ = predict_mean(p_opt, test_x, inputs=train_x, targets=train_y)
rmse = root_mean_square_error(predicted, target=test_y)
nll, _ = mll_cholesky(p_opt, inputs=test_x, targets=test_y)
print("A priori CG error:", cg_error)
print("A-priori RMSE:", rmse)
print("A-priori NLL:", nll)


print()

optimizer = jaxopt.LBFGS(
    value_and_grad, has_aux=True, value_and_grad=True, maxls=3, tol=0.1, verbose=False
)
optim_init = jax.jit(optimizer.init_state)
optim_update = jax.jit(optimizer.update)
state = optim_init(p_opt, subkey, inputs=train_x, targets=train_y)

progressbar = tqdm.tqdm(range(num_epochs))
progressbar.set_description(
    f"loss: {mll_train:.3F}, "
    f"test-nll: {nll:.3F}, "
    f"rmse: {rmse:.3F}, "
    f"cg_error: {cg_error:.3e}, "
)
start = time.perf_counter()

loss_timestamps = [0.0]
test_nlls = [nll]
test_rmses = [rmse]
loss_curve = [float(mll_train)]
cg_errors = [float(cg_error)]

for _ in progressbar:
    try:
        # Take the value and gradient
        # (value, aux), grads = value_and_grad(
        #     p_opt, subkey, inputs=train_x, targets=train_y
        # )
        # updates, state = optimizer.update(grads, state)
        # p_opt = optax.apply_updates(p_opt, updates)

        # Optimiser step
        key, subkey = jax.random.split(key, num=2)
        p_opt, state = optim_update(
            p_opt, state, subkey, inputs=train_x, targets=train_y
        )
        aux = state.aux
        value = state.value

        residual = aux["logpdf"]["residual"]
        cg_error = jnp.linalg.norm(residual) / jnp.sqrt(len(residual))

        # # Test NLL and RMSE
        predicted, _ = predict_mean(p_opt, test_x, inputs=train_x, targets=train_y)
        rmse = root_mean_square_error(predicted, target=test_y)
        nll, _ = mll_cholesky(p_opt, inputs=test_x, targets=test_y)
        print(unflatten(p_opt))

        # Save values
        current = time.perf_counter()
        loss_curve.append(float(value))
        cg_errors.append(float(cg_error))
        test_rmses.append(float(rmse))
        test_nlls.append(float(nll))
        loss_timestamps.append(current - start)
        progressbar.set_description(
            f"loss: {value:.3F}, "
            f"test-nll: {nll:.3F}, "
            f"rmse: {rmse:.3F}, "
            f"cg_error: {cg_error:.3e}, "
        )
    except KeyboardInterrupt:
        break
end = time.perf_counter()
print()

# Complete the data collection
loss_timestamps = jnp.asarray(loss_timestamps)
test_nlls = jnp.asarray(test_nlls)
test_rmses = jnp.asarray(test_rmses)
loss_curve = jnp.asarray(loss_curve)
cg_errors = jnp.asarray(cg_errors)


# Save results to a file
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)
jnp.save(f"{directory}loss_timestamps.npy", loss_timestamps)
jnp.save(f"{directory}test_nlls.npy", test_nlls)
jnp.save(f"{directory}test_rmses.npy", test_rmses)
jnp.save(f"{directory}loss_curve.npy", loss_curve)
jnp.save(f"{directory}cg_errors.npy", cg_errors)
