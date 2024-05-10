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
N = 50_000

# make train/val/test
n_train = int(0.8 * N)
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train:, :-1], data[n_train:, -1]

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
num_matvecs = 20
num_samples = 1
num_partitions = 10
rank_preconditioner = 1


# Set up linear algebra
v_like = jnp.ones((n_train,), dtype=float)
sample = hutchinson.sampler_normal(v_like, num=num_samples)
logdet = gp_util_linalg.krylov_logdet_slq(num_matvecs, sample=sample, num_batches=1)
solve_p = gp_util_linalg.krylov_solve_pcg_fixed_step(num_matvecs)
logpdf_fun = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)


# Set up a loss
cholesky_partial = gp_util_linalg.low_rank_cholesky_pivot(
    n_train, rank=rank_preconditioner
)
P = gp_util_linalg.precondition_low_rank(cholesky_partial, small_value=1e-4)
gram_matvec = gp_util_linalg.gram_matvec_map_over_batch(num_batches=num_partitions)
loss = gp_util.mll_exact_p(
    prior, likelihood, logpdf_p=logpdf_fun, gram_matvec=gram_matvec, precondition=P
)


p_opt, unflatten = jax.flatten_util.ravel_pytree([p_prior, p_likelihood])


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
print((t1 - t0) / 1)


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
print((t1 - t0) / 1)


progressbar = tqdm.tqdm(range(100))
error = jnp.linalg.norm(aux["residual"] / value, ord=jnp.inf)
progressbar.set_description(f"loss: {value:.3F}, cg_error: {error:.3F}")
start = time.perf_counter()

loss_curve = [float(value)]
loss_timestamps = [0.0]
cg_errors = [float(error)]

for _ in progressbar:
    try:
        key, subkey = jax.random.split(key, num=2)
        (value, aux), grads = value_and_grad_gp(
            p_opt, subkey, inputs=train_x, targets=train_y
        )

        error = jnp.linalg.norm(aux["residual"] / value, ord=jnp.inf)

        updates, state = optimizer.update(grads, state)
        p_opt = optax.apply_updates(p_opt, updates)
        progressbar.set_description(f"loss: {value:.3F}, cg_error: {error:.3F}")

        current = time.perf_counter()
        loss_curve.append(float(value))
        cg_errors.append(float(error))
        loss_timestamps.append(current - start)

    except KeyboardInterrupt:
        break
end = time.perf_counter()
opt_params = p_opt
