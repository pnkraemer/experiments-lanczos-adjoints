import os
import time

import jax
import jax.numpy as jnp
import optax
from jax_models.models.swin_transformer import *
from jax_models.models.van import *
from matfree_extensions.util import bnn_util, data_util, exp_util

seed = 0
rng = jax.random.PRNGKey(seed)

# Make directories
directory_fig = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory_fig, exist_ok=True)
directory_results = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory_results, exist_ok=True)


# Get Dataloader

train_loader = data_util.ImageNet1k_loaders(batch_size=128, seed=seed)

# Get model
model_rng, rng = jax.random.split(rng)
num_classes = 1000
# swin, params = load_model('convnext-base-224-1k', attach_head=True, num_classes=1000, dropout=0.0, pretrained=True)
model, params, batch_stats = van_tiny(pretrained=True, download_dir="weights/van")

model_apply = lambda p, x: model.apply(
    {"params": p, "batch_stats": batch_stats}, x, True
)
params_vec, unflatten = jax.flatten_util.ravel_pytree(params)
n_params = len(params_vec)

# GGN Function:  This was too slow! 2 hours for 1 gvp

# ggn_fun = bnn_util.ggn_vp_dataloader(params_vec, bnn_util.loss_training_cross_entropy_single,
#                                      model_fun=model_apply, param_unflatten=unflatten, data_loader=train_loader,)

# print(ggn_fun(jnp.ones_like(params_vec)))


# Log Determinant Function
lanczos_rank = 50
slq_num_samples = 1
slq_num_batches = 10
# logdet_fun = bnn_util.solver_logdet_slq_implicit(lanczos_rank=lanczos_rank, slq_num_samples=slq_num_samples, slq_num_batches=slq_num_batches)

# GGN vector product function
# ggn_fun = bnn_util.ggn_vp_parallel(loss_single=bnn_util.loss_training_cross_entropy_single,
#                                    model_fun=model_apply,
#                                    param_unflatten=unflatten)
# ggn_fun = bnn_util.kernel_vp_parallel(loss_single=bnn_util.loss_training_cross_entropy_single,
#                                       model_fun=model_apply,
#                                       param_unflatten=unflatten)

# ggn_fun = jax.jit(ggn_fun)


def unconstrain(a):
    return calibrate_log_alpha_min + jnp.exp(a)


# Callibration Loss

calib_rng, rng = jax.random.split(rng)
calib_loss = bnn_util.callibration_loss(model_apply, unflatten, unconstrain, n_params)
# calib_loss = jax.jit(calib_loss)
value_and_grad = jax.jit(jax.value_and_grad(calib_loss, argnums=0))

# Optimize alpha

calibrate_log_alpha_min = 0.1
calibrate_lrate = 1e-1
optimizer = optax.adam(calibrate_lrate)


alpha_rng, rng = jax.random.split(rng, num=2)
log_alpha = jax.random.normal(alpha_rng, shape=())
optimizer_state = optimizer.init(log_alpha)


# Epochs

for epoch, batch in enumerate(train_loader):
    model_rng, rng = jax.random.split(rng)
    img, label = batch["image"], batch["label"]
    img, label = jnp.asarray(img, dtype=float), jnp.asarray(label, dtype=float)
    start_time = time.perf_counter()
    loss, grad = value_and_grad(log_alpha, params_vec, img, label, rng)
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    log_alpha = optax.apply_updates(log_alpha, updates)
    print(
        f"Epoch: {epoch + 1}, loss {loss:.3f}, log alpha {log_alpha:.3f}, time {time.perf_counter() - start_time:.3f}"
    )

    if epoch == 50:
        break


breakpoint()

# Lanczos Parameters
numerics_lanczos_rank = 100
numerics_slq_num_samples = 10
numerics_slq_num_batches = 1
evaluate_num_samples = 5

logdet_fun = bnn_util.solver_logdet_slq_implicit(
    lanczos_rank=numerics_lanczos_rank,
    slq_num_samples=numerics_slq_num_samples,
    slq_num_batches=numerics_slq_num_batches,
)
sample_fun = bnn_util.sampler_lanczos(
    lanczos_rank=numerics_lanczos_rank, ggn_fun=ggn_fun, num=evaluate_num_samples
)
