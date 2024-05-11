import os.path
import time
import urllib.request

import jax
import jax.numpy as jnp
import optax
import scipy.io
import tqdm
from matfree import hutchinson
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
# Memory consumption: ~ num_data**2 * num_matvecs * num_samples / num_partitions
# todo: give cg fewer partitions than slq, because cg does not track batched samples!
num_data = 400
num_matvecs = 10
num_matvecs_cg_eval = 10  # todo: pass to mll_test (currently this arg is not used)
num_samples_batched = 10
num_samples_sequential = 1
num_partitions = 1
rank_precon = 1
small_value = 1e-10


memory_bytes = num_data**2 * num_matvecs * num_samples_batched / num_partitions * 32
memory_gb = memory_bytes / 8589934592
print(f"\nPredicting ~ {memory_gb} GB of memory\n")

# todo: shuffle?
data_sampled = data[:num_data, :-1], data[:num_data, -1]
train, test = data_util.split_train_test(*data_sampled, train=0.95)
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


# Set up a model
k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(3,), shape_out=())
prior = gp_util.model(gp_util.mean_zero(), k)
likelihood, p_likelihood = gp_util.likelihood_gaussian()


# Set up matrix-free linear algebra
# todo: why does solve_pcg_fixed_step_reortho nan out??
gram_matvec = gp_util_linalg.gram_matvec_map_over_batch(num_batches=num_partitions)
solve_p = gp_util_linalg.krylov_solve_pcg_fixed_step_reortho(num_matvecs)

# Initialise the parameters
key = jax.random.PRNGKey(1)
key, subkey = jax.random.split(key)
p_prior, p_likelihood = exp_util.tree_random_like(subkey, (p_prior, p_likelihood))
p_opt, unflatten = jax.flatten_util.ravel_pytree([p_prior, p_likelihood])


# Evaluate the loss function
@jax.jit
def mll_lanczos(params, *p_logdet, inputs, targets):
    p1, p2 = unflatten(params)

    # SLQ depends on the input size, so we define it here
    v_like = jnp.ones((len(inputs),), dtype=float)
    sample = hutchinson.sampler_rademacher(v_like, num=num_samples_batched)
    logdet = gp_util_linalg.krylov_logdet_slq(
        num_matvecs, sample=sample, num_batches=num_samples_sequential
    )

    # The preconditioner also depends on the inputs size
    low_rank = gp_util_linalg.low_rank_cholesky_pivot(len(inputs), rank_precon)
    precondition = gp_util_linalg.precondition_low_rank(
        low_rank, small_value=small_value
    )
    logpdf_p = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)

    # Build the loss and evaluate
    loss = gp_util.mll_exact_p(
        prior,
        likelihood,
        logpdf_p=logpdf_p,
        gram_matvec=gram_matvec,
        precondition=precondition,
    )

    val, info = loss(inputs, targets, *p_logdet, params_prior=p1, params_likelihood=p2)
    return -val, info


@jax.jit
def mll_cholesky(params, inputs, targets):
    p1, p2 = unflatten(params)
    logpdf = gp_util.logpdf_scipy_stats()

    # Build the loss and evaluate
    loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf, gram_matvec=gram_matvec)
    val, info = loss(inputs, targets, params_prior=p1, params_likelihood=p2)
    return -val, info


@jax.jit
def predict_mean(params, x, inputs, targets):
    p1, p2 = unflatten(params)

    # Use a Krylov solver with 2x as many steps
    solve = gp_util_linalg.krylov_solve_cg_fixed_step(2 * num_matvecs)
    posterior = gp_util.condition(
        prior, likelihood, gram_matvec=gram_matvec, solve=solve
    )
    return posterior(
        x, inputs=inputs, targets=targets, params_prior=p1, params_likelihood=p2
    )


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
optimizer = optax.adam(learning_rate=0.1)
state = optimizer.init(p_opt)
progressbar = tqdm.tqdm(range(50))
progressbar.set_description(
    f"loss: {mll_train:.3F}, "
    f"test-nll: {nll:.3F}, "
    f"rmse: {rmse:.3F}, "
    f"cg_error: {cg_error:.3e}"
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
        key, subkey = jax.random.split(key, num=2)
        (value, aux), grads = value_and_grad(
            p_opt, subkey, inputs=train_x, targets=train_y
        )
        residual = aux["logpdf"]["residual"]
        cg_error = jnp.linalg.norm(residual) / jnp.sqrt(len(residual))

        # Optimiser step
        updates, state = optimizer.update(grads, state)
        p_opt = optax.apply_updates(p_opt, updates)

        # # Test NLL and RMSE
        predicted, _ = predict_mean(p_opt, test_x, inputs=train_x, targets=train_y)
        rmse = root_mean_square_error(predicted, target=test_y)
        nll, _ = mll_cholesky(p_opt, inputs=test_x, targets=test_y)

        # Save values
        current = time.perf_counter()
        loss_curve.append(float(value))
        cg_errors.append(float(cg_error))
        test_rmses.append(float(rmse))
        test_nlls.append(float(nll))
        loss_timestamps.append(current - start)
        progressbar.set_description(
            f"loss: {mll_train:.3F}, "
            f"test-nll: {nll:.3F}, "
            f"rmse: {rmse:.3F}, "
            f"cg_error: {cg_error:.3e}"
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
