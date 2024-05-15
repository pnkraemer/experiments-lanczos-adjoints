import argparse
import os.path
import time

import jax
import jax.numpy as jnp
import optax
import tqdm
from matfree import hutchinson
from matfree_extensions import cg, low_rank
from matfree_extensions.util import data_util, exp_util, gp_util, uci_util


def load_data(which: str, /):
    if which == "concrete":
        return uci_util.uci_concrete()
    if which == "power_plant":
        return uci_util.uci_power_plant()
    if which == "parkinson":
        return uci_util.uci_parkinson()
    if which == "protein":
        return uci_util.uci_protein()
    if which == "bike_sharing":
        return uci_util.uci_bike_sharing()
    if which == "kegg_undirected":
        return uci_util.uci_kegg_undirected()
    if which == "kegg_directed":
        return uci_util.uci_kegg_directed()
    if which == "elevators":
        return uci_util.uci_elevators()
    if which == "kin40k":
        return uci_util.uci_kin40k()
    if which == "slice":
        return uci_util.uci_slice()
    raise ValueError


def root_mean_square_error(x, *, target):
    error_abs = x - target
    error = error_abs
    return jnp.sqrt(jnp.mean(error**2))


# Choose parameters
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--rank_precon", type=int, required=True)
parser.add_argument("--num_partitions", type=int, required=True)
parser.add_argument("--num_matvecs", type=int, required=True)
parser.add_argument("--num_samples", type=int, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--cg_tol", type=float, required=True)
args = parser.parse_args()
print(args)
# print("Arguments:", args)


# todo: we _could_ give CG a different matvec
#  (fewer partitions) to boost performance,
#  but it might not be necessary?
num_samples_sequential = args.num_samples
num_samples_batched = 1
num_matvecs_train_lanczos = args.num_matvecs
noise_minval = 1e-4
train_test_split = 0.8

# Load data
key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key, num=2)
inputs, targets = load_data(args.dataset)

# Subsample data to a multiple of num_partitions
num_data_raw = len(inputs)
coeff = num_data_raw // (5 * args.num_partitions)
num_data = int(coeff * 5 * args.num_partitions)
# print(
#     f"Subsampling data from N={num_data_raw} points "
#     f"to N={num_data} points to satisfy "
#     f"P={args.num_partitions} partitions "
#     f"and the train-test-split."
# )


# Predict memory consumption
memory_bytes = (
    num_data**2
    * num_samples_batched
    * num_matvecs_train_lanczos
    / args.num_partitions
    * 32
)
memory_gb = memory_bytes / 8589934592
# print(f"Predicting ~ {memory_gb} GB of memory")

# Train-test split

data_sampled = inputs[:num_data], targets[:num_data]
train, test = data_util.split_train_test_shuffle(
    subkey, *data_sampled, train=train_test_split
)
(train_x, train_y), (test_x, test_y) = train, test


# Set up linear algebra for training
solve_p = cg.pcg_adaptive(rtol=0.0, atol=args.cg_tol, maxiter=1000, miniter=10)
v_like = jnp.ones((len(train_x),), dtype=float)
sample = hutchinson.sampler_rademacher(v_like, num=num_samples_batched)
logdet = gp_util.krylov_logdet_slq(
    num_matvecs_train_lanczos,
    sample=sample,
    num_batches=num_samples_sequential,
    checkpoint=True,
)
rank_precon = int(jnp.minimum(args.rank_precon, len(train_x)))
cholesky = low_rank.cholesky_partial_pivot(rank=rank_precon)
precondition = low_rank.preconditioner(cholesky)
logpdf_p = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)
if args.num_partitions == 1:
    gram_matvec = gp_util.gram_matvec()
else:
    gram_matvec = gp_util.gram_matvec_partitioned(args.num_partitions, checkpoint=True)
constrain = gp_util.constraint_greater_than(noise_minval)
likelihood, p_likelihood = gp_util.likelihood_pdf_p(
    gram_matvec, logpdf_p, precondition=precondition, constrain=constrain
)

# Set up a model
ndim = train_x.shape[-1]
m, p_mean = gp_util.mean_constant(shape_out=())
k, p_kernel = gp_util.kernel_scaled_matern_32(shape_in=(ndim,), shape_out=())
prior = gp_util.model_gp(m, k)

# Build the loss and evaluate
loss = gp_util.target_logml(prior, likelihood)

# Initialise the parameters
ps = (p_mean, p_kernel, p_likelihood)
key, subkey = jax.random.split(key)
ps = exp_util.tree_random_like(subkey, ps)
p_opt, unflatten = jax.flatten_util.ravel_pytree(ps)


# Evaluate the loss function
@jax.jit
def mll_lanczos(params, *p_logdet, Xs, ys):
    p1, p2, p3 = unflatten(params)
    val, info = loss(
        Xs, ys, *p_logdet, params_mean=p1, params_kernel=p2, params_likelihood=p3
    )
    return -1.0 * val / len(Xs), info


@jax.jit
def mll_lanczos_eval(params, *p_args, Xs, ys):
    p1, p2, p3 = unflatten(params)

    solve_p = cg.pcg_adaptive(rtol=0.0, atol=1e-4, maxiter=10_000, miniter=10)
    v_like = jnp.ones((len(Xs),), dtype=float)
    sample = hutchinson.sampler_rademacher(v_like, num=num_samples_batched)
    logdet = gp_util.krylov_logdet_slq(
        num_matvecs_train_lanczos,
        sample=sample,
        num_batches=num_samples_sequential,
        checkpoint=True,
    )
    logpdf_p = gp_util.logpdf_krylov_p(solve_p=solve_p, logdet=logdet)
    lklhd, _ = gp_util.likelihood_pdf_p(
        gram_matvec, logpdf_p, precondition=precondition, constrain=constrain
    )

    loss_fun = gp_util.target_logml(prior, lklhd)
    val, info = loss_fun(
        Xs, ys, *p_args, params_mean=p1, params_kernel=p2, params_likelihood=p3
    )
    return -1.0 * val / len(Xs), info


@jax.jit
def predict_mean(params, x, Xs, ys):
    p1, p2, p3 = unflatten(params)

    solve_ = cg.pcg_adaptive(atol=1e-2, rtol=0.0, maxiter=10_000, miniter=10)
    likelihood_, _p_likelihood_ = gp_util.likelihood_condition_p(
        gram_matvec, solve_, precondition=precondition, constrain=constrain
    )
    posterior = gp_util.target_posterior(prior, likelihood_)

    postmean, _ = posterior(
        inputs=Xs, targets=ys, params_mean=p1, params_kernel=p2, params_likelihood=p3
    )
    return postmean(x)


optimizer = optax.adam(0.05)
state = optimizer.init(p_opt)
value_and_grad = jax.jit(jax.value_and_grad(mll_lanczos, argnums=0, has_aux=True))
# optimizer = jaxopt.LBFGS(value_and_grad, value_and_grad=True, has_aux=True)
# state = optimizer.init_state(p_opt, subkey, Xs=train_x, ys=train_y)


(value, aux), grads = value_and_grad(p_opt, subkey, Xs=train_x, ys=train_y)
noise = constrain(unflatten(p_opt)[-1]["raw_noise"])
progressbar = tqdm.tqdm(range(args.num_epochs))
residual = aux["logpdf"]["solve"]["residual_abs"]
cg_error = jnp.linalg.norm(residual) / jnp.sqrt(len(residual))
cg_numsteps = aux["logpdf"]["solve"]["num_steps"]
progressbar.set_description(
    f"loss: {value}, "
    # f"rmse: {None}, "
    f"cg_error: {cg_error}, "
    f"cg_numsteps: {cg_numsteps}, "
)

gradient_norms: dict = {}
for p_dict in unflatten(p_opt):
    for name, _value in p_dict.items():
        gradient_norms[name] = []

loss_timestamps = []
loss_curve = [float(value)]
cg_errors = [float(cg_error)]
cg_numsteps_all = [int(cg_numsteps)]
test_rmses: list = []
# slq_std_rels = [float(slq_std_rel)]

start = time.perf_counter()
for _ in progressbar:
    try:
        # Take the value and gradient
        key, subkey = jax.random.split(key, num=2)
        (value, aux), grads = value_and_grad(p_opt, subkey, Xs=train_x, ys=train_y)
        updates, state = optimizer.update(grads, state)
        p_opt = optax.apply_updates(p_opt, updates)
        # p_opt, state = optimizer.update(p_opt, state, subkey, Xs=train_x, ys=train_y)
        # value = state.value
        # aux = state.aux
        # grads = state.grad

        # Evaluate relevant quantities
        slq_std_rel = aux["logpdf"]["logdet"]["std_rel"]
        residual = aux["logpdf"]["solve"]["residual_abs"]
        cg_error = jnp.linalg.norm(residual) / jnp.sqrt(len(residual))
        cg_numsteps = aux["logpdf"]["solve"]["num_steps"]
        for p_dict in unflatten(grads):
            for name, gradval in p_dict.items():
                gradient_norms[name].append(jnp.linalg.norm(gradval))

        # predicted, predict_info = predict_mean(p_opt, test_x, Xs=train_x, ys=train_y)
        # rmse = root_mean_square_error(predicted, target=test_y)

        # Save values
        raw_noise = unflatten(p_opt)[-1]["raw_noise"]
        noise = constrain(raw_noise)
        current = time.perf_counter()
        # test_rmses.append(float(rmse))
        loss_curve.append(float(value))
        loss_timestamps.append(time.perf_counter() - start)
        cg_errors.append(float(cg_error))
        cg_numsteps_all.append(int(cg_numsteps))
        # slq_std_rels.append(float(slq_std_rel))
        progressbar.set_description(
            f"loss: {value:.3F}, "
            # f"rmse: {rmse:.3E}, "
            f"cg_error: {cg_error:.1e}, "
            f"cg_numsteps: {int(cg_numsteps)}, "
        )

    except KeyboardInterrupt:
        break
end = time.perf_counter()


# Complete the data collection
loss_curve = jnp.asarray(loss_curve)
loss_timestamps = jnp.asarray(loss_timestamps)
cg_errors = jnp.asarray(cg_errors)
cg_numsteps_all = jnp.asarray(cg_numsteps_all)


# Evaluate: RMSE & NLL
predicted, predict_info = predict_mean(p_opt, test_x, Xs=train_x, ys=train_y)
rmse = root_mean_square_error(predicted, target=test_y)

key, subkey = jax.random.split(key, num=2)
nll, _ = mll_lanczos_eval(p_opt, subkey, Xs=test_x, ys=test_y)
test_nlls = jnp.asarray(nll)
test_rmses = jnp.asarray(rmse)

print("NLL:", nll)
print("RMSE:", rmse)


# Save results to a file
directory = exp_util.matching_directory(__file__, "results/")
os.makedirs(directory, exist_ok=True)
path = f"{directory}{args.name}_{args.dataset}_s{args.seed}"
jnp.save(f"{path}_loss_timestamps.npy", loss_timestamps)
jnp.save(f"{path}_test_nlls.npy", test_nlls)
jnp.save(f"{path}_test_rmses.npy", test_rmses)
jnp.save(f"{path}_loss_curve.npy", loss_curve)
jnp.save(f"{path}_cg_errors.npy", cg_errors)
jnp.save(f"{path}_cg_numsteps_all.npy", cg_numsteps_all)
# jnp.save(f"{path}_slq_std_rels.npy", slq_std_rels)

for name, value in gradient_norms.items():
    jnp.save(f"{path}_gradient_norms_{name}.npy", jnp.asarray(value))

print()
print()
