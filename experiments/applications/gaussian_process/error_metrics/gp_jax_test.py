import argparse
import os.path
import time
import urllib.request

import jax
import jax.numpy as jnp
import optax
import scipy.io
import tqdm
from matfree import hutchinson
from matfree_extensions import low_rank
from matfree_extensions.util import data_util, exp_util, gp_util, gp_util_linalg


def root_mean_square_error(x, *, target):
    error_abs = x - target
    return jnp.sqrt(jnp.mean(error_abs**2))
    # return jnp.linalg.norm(error_abs) / jnp.sqrt(x.size)


if not os.path.isfile("../3droad.mat"):
    print("Downloading '3droad' UCI dataset...")
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1", "../3droad.mat"
    )

data = jnp.asarray(scipy.io.loadmat("../3droad.mat")["data"])
print(jnp.shape(data))

# Choose parameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--num_data", type=int, required=True)
parser.add_argument("--rank_precon", type=int, required=True)
parser.add_argument("--num_partitions", type=int, required=True)
parser.add_argument("--num_matvecs", type=int, required=True)
parser.add_argument("--num_samples", type=int, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
args = parser.parse_args()
print(args)


# todo: we _could_ give CG a different matvec
#  (fewer partitions) to boost performance,
#  but it might not be necessary?
num_samples_sequential = args.num_samples
num_matvecs_train_lanczos = args.num_matvecs
num_matvecs_train_cg = num_samples_sequential * args.num_matvecs
num_matvecs_eval_cg = 100 * num_matvecs_train_cg


memory_bytes = args.num_data**2 * num_matvecs_train_lanczos / args.num_partitions * 32
memory_gb = memory_bytes / 8589934592
print(f"\nPredicting ~ {memory_gb} GB of memory\n")

key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key, num=2)
data_sampled = data[: args.num_data, :-1], data[: args.num_data, -1]
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
sample = hutchinson.sampler_rademacher(v_like, num=1)
logdet = gp_util_linalg.krylov_logdet_slq(
    num_matvecs_train_lanczos, sample=sample, num_batches=num_samples_sequential
)
cholesky = low_rank.cholesky_partial_pivot(rank=args.rank_precon)
precondition = low_rank.preconditioner(cholesky)
logpdf_p = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)
gram_matvec = gp_util_linalg.gram_matvec_partitioned(args.num_partitions)
likelihood, p_likelihood = gp_util.likelihood_pdf_p(gram_matvec, logpdf_p, precondition)

# Set up a model
m, p_mean = gp_util.mean_constant(shape_out=())
k, p_kernel = gp_util.kernel_scaled_matern_32(shape_in=(3,), shape_out=())
prior = gp_util.model_gp(m, k)

# Build the loss and evaluate
loss = gp_util.target_logml(prior, likelihood)

# Initialise the parameters
key, subkey = jax.random.split(key)
ps = (p_mean, p_kernel, p_likelihood)
ps = exp_util.tree_random_like(subkey, ps)
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
    return -val / len(inputs), info


@jax.jit
def mll_cholesky(params, inputs, targets):
    p1, p2, p3 = unflatten(params)

    # Build the loss and evaluate
    logpdf = gp_util.logpdf_scipy_stats()
    lklhd, _ = gp_util.likelihood_pdf(gram_matvec, logpdf)

    loss_fun = gp_util.target_logml(prior, lklhd)
    val, info = loss_fun(
        inputs, targets, params_mean=p1, params_kernel=p2, params_likelihood=p3
    )
    return -val / len(inputs), info


# Use a Krylov solver with 2x as many steps for evaluation
solve = gp_util_linalg.krylov_solve_cg_fixed_step(num_matvecs_eval_cg)
likelihood_, _p_likelihood_ = gp_util.likelihood_condition(gram_matvec, solve)

posterior = gp_util.target_posterior(prior, likelihood_)


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
slq_std_rel = aux["logpdf"]["logdet"]["std_rel"]
residual = aux["logpdf"]["solve"]["residual"]
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
print("A-priori CG error:", cg_error)
print("A-priori SLQ std (rel):", slq_std_rel)
print("A-priori RMSE:", rmse)
print("A-priori NLL:", nll)


print()

optimizer = optax.adam(0.1)
state = optimizer.init(p_opt)

progressbar = tqdm.tqdm(range(args.num_epochs))
progressbar.set_description(
    f"loss: {mll_train:.3F}, "
    f"test-nll: {nll:.3F}, "
    f"rmse: {rmse:.3F}, "
    f"cg_error: {cg_error:.1e}, "
    f"slq_std_rel: {slq_std_rel:.1e}, "
)
start = time.perf_counter()

loss_timestamps = [0.0]
test_nlls = [nll]
test_rmses = [rmse]
loss_curve = [float(mll_train)]
cg_errors = [float(cg_error)]
slq_std_rels = [float(slq_std_rel)]

for _ in progressbar:
    try:
        # Take the value and gradient
        key, subkey = jax.random.split(key, num=2)
        (value, aux), grads = value_and_grad(
            p_opt, subkey, inputs=train_x, targets=train_y
        )
        updates, state = optimizer.update(grads, state)
        p_opt = optax.apply_updates(p_opt, updates)
        print("grad", unflatten(grads))

        # Optimiser step
        slq_std_rel = aux["logpdf"]["logdet"]["std_rel"]
        residual = aux["logpdf"]["solve"]["residual"]
        cg_error = jnp.linalg.norm(residual) / jnp.sqrt(len(residual))

        # # Test NLL and RMSE
        predicted, _ = predict_mean(p_opt, test_x, inputs=train_x, targets=train_y)
        rmse = root_mean_square_error(predicted, target=test_y)
        nll, _ = mll_cholesky(p_opt, inputs=test_x, targets=test_y)
        # print(state)
        # print()
        print(unflatten(p_opt)[-1]["raw_noise"])
        print()

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
            f"cg_error: {cg_error:.1e}, "
            f"slq_std_rel: {slq_std_rel:.1e} "
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
