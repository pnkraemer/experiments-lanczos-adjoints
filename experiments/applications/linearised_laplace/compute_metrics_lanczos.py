import os
import pickle
import time

import jax
import jax.numpy as jnp
import optax
from jax_models.models.van import van_tiny
from matfree_extensions.util import bnn_util, data_util, exp_util, bnn_baselines
from sklearn.metrics import roc_auc_score
import numpy as np

seed = 2
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

val_loader = data_util.get_imagenet_test_loader(
     batch_size=50, seed=seed, n_samples_per_class=0.1
)

ood_loader = data_util.get_places365(batch_size=50, seed=seed)
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

def model_linear(sample, v, x):
    """Evaluate the model after linearising around the optimal parameters."""
    fx = model_apply(unflatten(v), x)
    _, jvp = jax.jvp(lambda p: model_apply(unflatten(p), x), (v,), (sample - v,))
    return fx + jvp

def auroc(scores_id, scores_ood):
    labels = np.zeros(len(scores_id) + len(scores_ood), dtype="int32")
    labels[len(scores_id) :] = 1
    scores = np.concatenate([scores_id, scores_ood])
    return roc_auc_score(labels, scores)


calibrate_log_alpha_min = 0.1

def unconstrain(a):
    # return calibrate_log_alpha_min + _softplus(a)
    return calibrate_log_alpha_min + jnp.exp(a)

ggn_fun = bnn_util.ggn_vp_parallel(
    loss_single=bnn_util.loss_training_cross_entropy_single,
    model_fun=model_apply,
    param_unflatten=unflatten,
)
batch = next(iter(train_loader))
x_train, y_train = jnp.asarray(batch["image"], dtype=float), jnp.asarray(batch["label"], dtype=float)

test_batch = next(iter(val_loader))
x_test, y_test = jnp.asarray(test_batch["image"], dtype=float), jnp.asarray(test_batch["label"], dtype=float)

ood_batch = next(iter(ood_loader))
x_ood, y_ood = jnp.asarray(ood_batch["image"], dtype=float), jnp.asarray(ood_batch["label"], dtype=float)

key = jax.random.PRNGKey(0)
num_samples = 10
lanczos_rank = 10


log_alpha_full = 0.43
ggn_vp_full = lambda v: jax.flatten_util.ravel_pytree(ggn_fun(v, params_vec, x_train, y_train))[0] + unconstrain(log_alpha_full) * v
posterior_samples = bnn_util.lanczos_sampler(ggn_vp=ggn_vp_full, num_samples = num_samples, lanczos_rank=lanczos_rank, key=key, params_vec=params_vec)

def sample_jvp(sample, v, x):
    _, jvp = jax.jvp(lambda p: model_apply(unflatten(p), x), (v,), (sample - v,))
    return jnp.linalg.norm(jvp)

sample_norms = jax.vmap(lambda v: sample_jvp(v, params_vec, x_train))(posterior_samples)
print("Sample norms:", sample_norms)

logits = jax.vmap(lambda v: model_linear(v, params_vec, x_train))(posterior_samples)
logprobs = jnp.mean(jax.nn.log_softmax(logits), axis=0)
sample_nll = jnp.sum(y_train * logprobs)
print("Posterior NLL:", sample_nll)

logits_test = jax.vmap(lambda v: model_linear(v, params_vec, x_test))(posterior_samples)
logprobs_test = jnp.mean(jax.nn.log_softmax(logits_test), axis=0)
sample_test_nll = jnp.sum(y_test * logprobs_test)
print("Posterior Test NLL:", sample_test_nll)

posterior_ece = bnn_util.metric_ece(probs=jnp.mean(jax.nn.softmax(logits_test, axis=-1),axis=0), labels_hot=y_train, num_bins=10)[0]
print("Posterior ECE:", posterior_ece)
ood_logits = jax.vmap(lambda v: model_linear(v, params_vec, x_ood))(posterior_samples)
score_id = jnp.std(jax.nn.softmax(logits_test, axis=-1), axis=0).min(axis=-1)
score_ood = jnp.std(jax.nn.softmax(ood_logits, axis=-1), axis=0).min(axis=-1)
auroc_posterior = auroc(score_id, score_ood)
print("posterior AUROC:", auroc_posterior)

print()


map_logits = model_apply(params, x_train)
map_logprobs = jax.nn.log_softmax(map_logits)
map_nll = jnp.sum(y_train * map_logprobs)
print("MAP NLL:", map_nll)
print("MAP ECE:", bnn_util.metric_ece(probs=jax.nn.softmax(map_logits, axis=-1), labels_hot=y_train, num_bins=10)[0])

map_test_logits = model_apply(params, x_test)
map_test_logprobs = jax.nn.log_softmax(map_test_logits)
map_test_nll = jnp.sum(y_test * map_test_logprobs)
print("MAP NLL:", map_test_nll)

print()


log_alpha_diag = 1.73
ggn_vp_diag = lambda v: jax.flatten_util.ravel_pytree(ggn_fun(v, params_vec, x_train, y_train))[0] + (unconstrain(log_alpha_diag) * v)
diag_samples = bnn_util.lanczos_sampler(ggn_vp=ggn_vp_diag, num_samples = num_samples, lanczos_rank=lanczos_rank, key=key, params_vec=params_vec)
logits = jax.vmap(lambda v: model_linear(v, params_vec, x_train))(diag_samples)
logprobs_diag_lanczos = jnp.mean(jax.nn.log_softmax(logits), axis=0)

sample_diag_nll_lanczos = jnp.sum(y_train * logprobs_diag_lanczos)
print("Posterior Diag NLL lanczos sample:", sample_diag_nll_lanczos)
print()
logits_test = jax.vmap(lambda v: model_linear(v, params_vec, x_test))(diag_samples)
logprobs_diag_test_lanczos = jnp.mean(jax.nn.log_softmax(logits_test), axis=0)
sample_diag_test_nll_lanczos = jnp.sum(y_test * logprobs_diag_test_lanczos)
print("Posterior Diag Test NLL lanczos sample:", sample_diag_test_nll_lanczos)

diag_lanczos_ece = bnn_util.metric_ece(probs=jnp.mean(jax.nn.softmax(logits_test, axis=-1),axis=0), labels_hot=y_train, num_bins=10)[0]
print("diag lanczos ECE:", diag_lanczos_ece)
ood_logits = jax.vmap(lambda v: model_linear(v, params_vec, x_ood))(diag_samples)
score_id = jnp.std(jax.nn.softmax(logits_test, axis=-1), axis=0).min(axis=-1)
score_ood = jnp.std(jax.nn.softmax(ood_logits, axis=-1), axis=0).min(axis=-1)
diag_lanczos_auroc = auroc(score_id, score_ood)
print("diag lanczos AUROC:", diag_lanczos_auroc)

print()


gvp_fn_batch = jax.tree_util.Partial(
    ggn_fun, params_vec=params_vec, x_batch=x_train, y_batch=y_train
)
key = jax.random.PRNGKey(0)
diagonal = bnn_baselines.hutchinson_diagonal(gvp_fn_batch, unflatten(params_vec), n_samples=25, key=key, num_levels=3, computation_type="serial")
diagonal = jax.flatten_util.ravel_pytree(diagonal)[0] 
diagonal = jnp.where(diagonal < 1e-8, 0.0, diagonal)
diagonal_diag = diagonal + unconstrain(log_alpha_diag)
eps = jax.random.normal(key, (num_samples, *params_vec.shape))
diag_samples = jax.vmap(lambda e: params_vec + e * 1/jnp.sqrt(diagonal_diag))(eps)
# diag_samples = bnn_util.lanczos_sampler(ggn_vp=ggn_vp_diag, num_samples = num_samples, lanczos_rank=lanczos_rank, key=key, params_vec=params_vec)
logits = jax.vmap(lambda v: model_linear(v, params_vec, x_train))(diag_samples)
logprobs_diag = jnp.mean(jax.nn.log_softmax(logits), axis=0)
sample_diag_nll = jnp.sum(y_train * logprobs_diag)
print("Posterior Diag NLL diag sample:", sample_diag_nll)
print()
logits_test = jax.vmap(lambda v: model_linear(v, params_vec, x_test))(diag_samples)
logprobs_diag_test = jnp.mean(jax.nn.log_softmax(logits_test), axis=0)
sample_diag_test_nll = jnp.sum(y_test * logprobs_diag_test)
print("Posterior Diag Test NLL diag sample:", sample_diag_test_nll)

diag_ece = bnn_util.metric_ece(probs=jnp.mean(jax.nn.softmax(logits_test, axis=-1),axis=0), labels_hot=y_train, num_bins=10)[0]
print("diag ECE:", diag_ece)
ood_logits = jax.vmap(lambda v: model_linear(v, params_vec, x_ood))(diag_samples)
score_id = jnp.std(jax.nn.softmax(logits_test, axis=-1), axis=0).min(axis=-1)
score_ood = jnp.std(jax.nn.softmax(ood_logits, axis=-1), axis=0).min(axis=-1)
diag_auroc = auroc(score_id, score_ood)
print("diag AUROC:", diag_auroc)

print()

# log_alpha_uncallibrated = jax.random.normal(key, shape=()) * 0.1
# ggn_vp_uncallibrated = lambda v: jax.flatten_util.ravel_pytree(ggn_fun(v, params_vec, x_train, y_train))[0] + unconstrain(log_alpha_uncallibrated) * v
# uncallibrated_samples = bnn_util.lanczos_sampler(ggn_vp=ggn_vp_uncallibrated, num_samples = num_samples, lanczos_rank=lanczos_rank, key=key, params_vec=params_vec)
# logits = jax.vmap(lambda v: model_linear(v, params_vec, x_train))(uncallibrated_samples)
# logprobs_uncallibrated = jnp.mean(jax.nn.log_softmax(logits), axis=0)
# sample_uncallibrated_nll = jnp.sum(y_train * logprobs_uncallibrated)
# print("Posterior Uncallibrated NLL diag sample:", sample_uncallibrated_nll)
# print()
# logits_test = jax.vmap(lambda v: model_linear(v, params_vec, x_test))(uncallibrated_samples)
# logprobs_uncallibrated_test = jnp.mean(jax.nn.log_softmax(logits_test), axis=0)
# sample_uncallibrated_test_nll = jnp.sum(y_test * logprobs_uncallibrated_test)
# print("Posterior Uncallibrated Test NLL diag sample:", sample_uncallibrated_test_nll)
# print()

# diagonal_uncallibrated = diagonal + unconstrain(log_alpha_uncallibrated)
# eps = jax.random.normal(key, (num_samples, *params_vec.shape))
# diag_samples = jax.vmap(lambda e: params_vec + e * 1/jnp.sqrt(diagonal_uncallibrated))(eps)
# logits = jax.vmap(lambda v: model_linear(v, params_vec, x_train))(diag_samples)
# logprobs_diag_uncalib = jnp.mean(jax.nn.log_softmax(logits), axis=0)
# sample_diag_nll_uncalib = jnp.sum(y_train * logprobs_diag_uncalib)
# print("Posterior Diag NLL diag sample:", sample_diag_nll_uncalib)
# print()
# logits_test = jax.vmap(lambda v: model_linear(v, params_vec, x_test))(diag_samples)
# logprobs_diag_test_uncalib = jnp.mean(jax.nn.log_softmax(logits_test), axis=0)
# sample_diag_test_nll_uncalib = jnp.sum(y_test * logprobs_diag_test_uncalib)
# print("Posterior Diag Test NLL diag sample:", sample_diag_test_nll_uncalib)
# print()


results = {
    "sample_nll": sample_nll,
    "sample_test_nll": sample_test_nll,
    "posterior_ece": posterior_ece,
    "auroc_posterior": auroc_posterior,
    "map_nll": map_nll,
    "map_test_nll": map_test_nll,
    "sample_diag_nll_lanczos": sample_diag_nll_lanczos,
    "sample_diag_test_nll_lanczos": sample_diag_test_nll_lanczos,
    "diag_ece_laczos": diag_lanczos_ece,
    "diag_auroc_lanczos": diag_lanczos_auroc,
    "sample_diag_nll": sample_diag_nll,
    "sample_diag_test_nll": sample_diag_test_nll,
    "diag_ece": diag_ece,
    "diag_auroc": diag_auroc,

    # "sample_uncallibrated_nll": sample_uncallibrated_nll,
    # "sample_uncallibrated_test_nll": sample_uncallibrated_test_nll,
    # "sample_diag_nll_uncalib": sample_diag_nll_uncalib,
    # "sample_diag_test_nll_uncalib": sample_diag_test_nll_uncalib,
    }

save_path = "./results/applications/linearised_laplace/all_eval_metrics_lanczos_2"
pickle.dump(results, open(f"{save_path}.pickle", "wb"))




