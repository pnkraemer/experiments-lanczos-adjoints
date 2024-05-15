import os
import pickle
import time

import jax
import jax.numpy as jnp
import optax
from jax_models.models.van import van_tiny
from matfree_extensions.util import bnn_util, data_util, exp_util

directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)

seed = 0
rng = jax.random.PRNGKey(seed)

train_loader = data_util.ImageNet1k_loaders(
    batch_size=50, seed=seed, n_samples_per_class=0.1
)

model_rng, rng = jax.random.split(rng)
num_classes = 1000
model, params, batch_stats = van_tiny(pretrained=True, download_dir="weights/van")

model_apply = lambda p, x: model.apply(
    {"params": p, "batch_stats": batch_stats}, x, True
)
params_vec, unflatten = jax.flatten_util.ravel_pytree(params)
n_params = len(params_vec)
print(f"Number of parameters: {n_params}")

lanczos_rank = 50
slq_num_samples = 1
slq_num_batches = 10

calibrate_log_alpha_min = 0.1
calibrate_lrate = 1e-2
optimizer = optax.adam(calibrate_lrate)
log_alpha_init = -0.5
optimizer_state = optimizer.init(log_alpha_init)


def unconstrain(a):
    return calibrate_log_alpha_min + jnp.exp(a)


log_alpha_list = jnp.array([0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
alpha_list = jax.vmap(unconstrain)(log_alpha_list)
calib_rng, rng = jax.random.split(rng)
calib_loss = bnn_util.callibration_loss(model_apply, unflatten, unconstrain, n_params)
value_and_grad = jax.jit(jax.value_and_grad(calib_loss, argnums=0))


# batch2 = next(iter(train_loader))
# label2 = jnp.asarray(batch["label"], dtype=float)
# img2 = jnp.asarray(batch["image"], dtype=float)


calib_loss_list = []

start_time = time.perf_counter()
for log_alpha in log_alpha_list:
    for batch in train_loader:
        label = jnp.asarray(batch["label"], dtype=float)
        img = jnp.asarray(batch["image"], dtype=float)
        log_marg = calib_loss(log_alpha, params_vec, img, label, rng)
        calib_loss_list.append(log_marg)
        print(f"alpha: {unconstrain(log_alpha)}, loss: {log_marg}")

results = {
    "log_alphas": log_alpha_list,
    "losses": calib_loss_list,
    "time": time.perf_counter() - start_time,
}
save_path = "./results/applications/linearised_laplace/imagenet_callibration_gridsearch"
pickle.dump(results, open(f"{save_path}.pickle", "wb"))
