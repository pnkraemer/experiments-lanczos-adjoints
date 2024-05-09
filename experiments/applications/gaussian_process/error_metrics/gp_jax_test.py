import os.path
import time
import urllib.request

import jax
import jax.numpy as jnp
import optax
import tqdm
from matfree import hutchinson
from matfree_extensions.util import gp_util
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


k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(3,), shape_out=())
likelihood, p_likelihood = gp_util.likelihood_gaussian()

# Set up all constant parameters
gram_matvec = gp_util.gram_matvec_full_batch()
v_like = jnp.ones((n_train,), dtype=float)
sample = hutchinson.sampler_normal(v_like, num=1)
logdet = gp_util.krylov_logdet_slq(10, sample=sample, num_batches=5)

# [PRECON] Set up a solver
solve = gp_util.krylov_solve_cg_precondition(tol=1.0, maxiter=100)
logpdf_fun = gp_util.logpdf_krylov(solve=solve, logdet=logdet)

# [PRECON] Set up a GP model
low_rank_impl = gp_util.low_rank_cholesky_pivot(n_train, rank=1)
P = gp_util.precondition_low_rank(low_rank_impl, small_value=1e-4)
prior = gp_util.model_precondition(
    gp_util.mean_zero(), k, gram_matvec=gram_matvec, precondition=P
)
loss = gp_util.mll_exact(prior, likelihood, logpdf=logpdf_fun)

key = jax.random.PRNGKey(1)
p_opt, unflatten = jax.flatten_util.ravel_pytree([p_prior, p_likelihood])
optimizer = optax.adam(learning_rate=0.1)
state = optimizer.init(p_opt)


def mll_lanczos(params, key, inputs, targets):
    p1, p2 = unflatten(params)
    val, info = loss(inputs, targets, key, params_prior=p1, params_likelihood=p2)
    return -val, info


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
progressbar.set_description(f"loss: {value:.3F}")
start = time.perf_counter()

loss_curve = [float(value)]
loss_timestamps = [0.0]

for _ in progressbar:
    try:
        key, subkey = jax.random.split(key, num=2)
        (value, aux), grads = value_and_grad_gp(
            p_opt, subkey, inputs=train_x, targets=train_y
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
