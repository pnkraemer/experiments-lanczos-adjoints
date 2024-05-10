import os
import pickle
import time

import jax
import jax.numpy as jnp
import optax
from jax_models.models.van import van_tiny
from matfree_extensions.util import bnn_util, data_util, exp_util

seed = 0
rng = jax.random.PRNGKey(seed)

# Make directories
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)


# Get Dataloader

train_loader = data_util.ImageNet1k_loaders(
    batch_size=50, seed=seed, n_samples_per_class=0.1
)

# Get model
model_rng, rng = jax.random.split(rng)
num_classes = 1000
model, params, batch_stats = van_tiny(pretrained=True, download_dir="weights/van")

model_apply = lambda p, x: model.apply(
    {"params": p, "batch_stats": batch_stats}, x, True
)
params_vec, unflatten = jax.flatten_util.ravel_pytree(params)
n_params = len(params_vec)
print(f"Number of parameters: {n_params}")

# Log Determinant Function
lanczos_rank = 50
slq_num_samples = 1
slq_num_batches = 10

def _softplus(x, beta=1.0, threshold=20.0):
    x_safe = jax.lax.select(x * beta < threshold, x, jax.numpy.ones_like(x))
    return jax.lax.select(
        x * beta < threshold,
        1 / beta * jax.numpy.log(1 + jax.numpy.exp(beta * x_safe)),
        x,
    )


def unconstrain(a):
    # return calibrate_log_alpha_min + _softplus(a)
    return calibrate_log_alpha_min + jnp.exp(a)


# Callibration Loss

calib_rng, rng = jax.random.split(rng)
calib_loss = bnn_util.callibration_loss(model_apply, unflatten, unconstrain, n_params)
# calib_loss = jax.jit(calib_loss)
value_and_grad = jax.jit(jax.value_and_grad(calib_loss, argnums=0))

# Optimize alpha

calibrate_log_alpha_min = 0.1
calibrate_lrate = 1e-2
optimizer = optax.rmsprop(calibrate_lrate)
optimizer = optax.MultiSteps(optimizer, every_k_schedule=5)

def build_train_step(optimizer: optax.GradientTransformation):
  """Builds a function for executing a single step in the optimization."""

  @jax.jit
  def update(log_alpha, params_vec, opt_state, img, label, rng):
    loss, grad = value_and_grad(log_alpha, params_vec, img, label, rng)
    updates, opt_state = optimizer.update(grad, opt_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    return log_alpha, opt_state, loss

  return update

alpha_rng, rng = jax.random.split(rng, num=2)
log_alpha = jax.random.normal(alpha_rng, shape=())


train_step = build_train_step(optimizer)
optimizer_state = optimizer.init(log_alpha)

# Epochs
log_alphas = []
losses = []
n_epochs = 10
for epoch in range(n_epochs):
    for i, batch in enumerate(train_loader):
    # i = epoch
        model_rng, rng = jax.random.split(rng)
        img, label = batch["image"], batch["label"]
        img, label = jnp.asarray(img, dtype=float), jnp.asarray(label, dtype=float)
        start_time = time.perf_counter()
        log_alpha, optimizer_state, loss = train_step(log_alpha, params_vec, optimizer_state, img, label, rng)
        # updates, optimizer_state = optimizer.update(grad, optimizer_state)
        # log_alpha = optax.apply_updates(log_alpha, updates)
        end_time = time.perf_counter()
        print(
            f"iter: {i + 1}, loss {loss:.3f}, alpha {unconstrain(log_alpha):.3f}, time {end_time - start_time:.3f}"
        )
        log_alphas.append(log_alpha)
        losses.append(loss)

results = {"log_alphas": log_alphas, "losses": losses}
save_path = "./results/applications/linearised_laplace/imagenet_callibration"
pickle.dump(results, open(f"{save_path}.pickle", "wb"))
