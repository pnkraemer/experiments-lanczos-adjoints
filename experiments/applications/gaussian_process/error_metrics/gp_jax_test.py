import os.path
import time
import urllib.request

import jax
import jax.numpy as jnp
import optax
import tqdm
from matfree import hutchinson
from matfree_extensions.util import gp_util, gp_util_linalg
from scipy.io import loadmat

if not os.path.isfile("../3droad.mat"):
    print("Downloading '3droad' UCI dataset...")
    urllib.request.urlretrieve(
        "https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1", "../3droad.mat"
    )

data = jnp.asarray(loadmat("../3droad.mat")["data"])

N = data.shape[0]
N = 100_000

# make train/val/test
n_train = int(0.8 * N)
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train:N, :-1], data[n_train:N, -1]

# normalize features
mean = train_x.mean(axis=-2, keepdims=True)
std = train_x.std(axis=-2, keepdims=True) + 1e-6  # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# normalize labels
mean, std = train_y.mean(), train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std



# Set up a model
k, p_prior = gp_util.kernel_scaled_matern_32(shape_in=(3,), shape_out=())
prior = gp_util.model(gp_util.mean_zero(), k)
likelihood, p_likelihood = gp_util.likelihood_gaussian()


# Choose parameters
num_matvecs = 10
num_samples = 1
num_partitions = 50


# Set up linear algebra
v_like = jnp.ones((n_train,), dtype=float)
sample = hutchinson.sampler_normal(v_like, num=num_samples)
logdet = gp_util_linalg.krylov_logdet_slq(num_matvecs, sample=sample, num_batches=1)
solve = gp_util_linalg.krylov_solve_cg_fixed_step_reortho(num_matvecs)
logpdf_fun = gp_util.logpdf_krylov(solve=solve, logdet=logdet)


# Set up a loss
gram_matvec = gp_util_linalg.gram_matvec_map_over_batch(num_batches=num_partitions)
loss = gp_util.mll_exact(
    prior, likelihood, logpdf=logpdf_fun, gram_matvec=gram_matvec
)


p_opt, unflatten = jax.flatten_util.ravel_pytree([p_prior, p_likelihood])

@jax.jit
def mll_lanczos(params, key, inputs, targets):
    p1, p2 = unflatten(params)
    val, info = loss(inputs, targets, key, params_prior=p1, params_likelihood=p2)
    return -val, info


key = jax.random.PRNGKey(1)
optimizer = optax.adam(learning_rate=0.1)
state = optimizer.init(p_opt)

fun = jax.jit(mll_lanczos)
(value, aux) = fun(p_opt, key, inputs=train_x, targets=train_y)
value.block_until_ready()


t0 = time.perf_counter()
for _ in range(1):
    (value, aux) = fun(p_opt, key, inputs=train_x, targets=train_y)
    value.block_until_ready()
t1 = time.perf_counter()
print("Runtie value:", (t1 - t0) / 1)


value_and_grad_gp = jax.jit(jax.value_and_grad(mll_lanczos, argnums=0, has_aux=True))
(value, aux), grad = value_and_grad_gp(p_opt, key, inputs=train_x, targets=train_y)
value.block_until_ready()
grad.block_until_ready()


t0 = time.perf_counter()
for _ in range(1):
    (value, aux), grad = value_and_grad_gp(p_opt, key, inputs=train_x, targets=train_y)
    value.block_until_ready()
    grad.block_until_ready()
t1 = time.perf_counter()
print("RUntime value and Grad:", (t1 - t0) / 1)


# Test loss (double compute budget compared to training)
v_like = jnp.ones((len(test_x),), dtype=float)
sample = hutchinson.sampler_normal(v_like, num=2*num_samples)
logdet = gp_util_linalg.krylov_logdet_slq(2*num_matvecs, sample=sample, num_batches=2*1)
solve = gp_util_linalg.krylov_solve_cg_fixed_step_reortho(2*num_matvecs)
logpdf_fun = gp_util.logpdf_krylov(solve=solve, logdet=logdet)
gram_matvec = gp_util_linalg.gram_matvec_map_over_batch(num_batches=num_partitions)
loss_test = gp_util.mll_exact(
    prior, likelihood, logpdf=logpdf_fun, gram_matvec=gram_matvec)

@jax.jit
def mll_lanczos_test(params, key, inputs, targets):
    p1, p2 = unflatten(params)
    val, info = loss_test(inputs, targets, key, params_prior=p1, params_likelihood=p2)
    return -val, info


@jax.jit
def rmse_test(params, train_inputs, train_targets, test_inputs, test_targets):
    p1, p2 = unflatten(params)
    mean, kernel_prior = prior(train_inputs, params=p1)
    mean_, kernel_likelihood = likelihood(mean, kernel_prior, params=p2)

    # Build matvec for likelihood

    def cov_matvec_likelihood(v):
        cov = gram_matvec(kernel_likelihood)
        idx = jnp.arange(len(train_inputs))
        return cov(idx, idx, v)


    K_inv_times_y, _info = solve(cov_matvec_likelihood, train_targets)

    # Build matvec for prior

    mean, kernel_prior = prior(test_inputs, params=p1)
    mean_, kernel_likelihood = likelihood(mean, kernel_prior, params=p2)


    def cov_matvec_prior(v):
        cov = gram_matvec(kernel_prior)
        idx = jnp.arange(len(train_inputs))
        idy = jnp.arange(len(test_inputs))
        return cov(idy, idx, v)

    reconstruction = cov_matvec_prior(K_inv_times_y)
    return jnp.linalg.norm(reconstruction - test_targets)/jnp.sqrt(len(test_targets))


rmse = rmse_test(p_opt, train_x, train_y, test_x, test_y)
print("test rmse", rmse)



key, subkey = jax.random.split(key, num=2)
(test_nll, _aux) = mll_lanczos_test(p_opt, key, inputs=test_x, targets=test_y)

progressbar = tqdm.tqdm(range(50))
error = jnp.linalg.norm(aux["residual"] / value, ord=jnp.inf)
progressbar.set_description(f"loss: {value:.3F}, test-nll: {test_nll:.3F}, cg_error: {error:.3e}")
start = time.perf_counter()

loss_curve = [float(value)]
loss_timestamps = [0.0]
test_nlls = [test_nll]
test_rmses = [rmse]
cg_errors = [float(error)]

for _ in progressbar:
    try:

        # value and grad
        key, subkey = jax.random.split(key, num=2)
        (value, aux), grads = value_and_grad_gp(
            p_opt, subkey, inputs=train_x, targets=train_y
        )
        cg_error = jnp.linalg.norm(aux["residual"] / (1e-6 + value), ord=jnp.inf)

        # Optimiser step
        updates, state = optimizer.update(grads, state)
        p_opt = optax.apply_updates(p_opt, updates)

        # Test NLL
        key, subkey = jax.random.split(key, num=2)
        (test_nll, _aux) = mll_lanczos_test(p_opt, key, inputs=test_x, targets=test_y)

        # Test RMSE
        rmse = rmse_test(p_opt, train_x, train_y, test_x, test_y)

        # Save values
        current = time.perf_counter()
        loss_curve.append(float(value))
        cg_errors.append(float(cg_error))
        test_rmses.append(float(rmse))
        test_nlls.append(float(test_nll))
        loss_timestamps.append(current - start)
        progressbar.set_description(f"loss: {value:.3F}, test-nll: {test_nll:.3f}, test-rmse {rmse:.3f}, cg_error: {cg_error:.3e}")

    except KeyboardInterrupt:
        break
end = time.perf_counter()
opt_params = p_opt
