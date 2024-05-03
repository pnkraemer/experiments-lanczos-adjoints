import os
import time

import jax
import jax.experimental.sparse
import jax.numpy as jnp
import tqdm
from matfree import hutchinson
from matfree_extensions import exp_util
from matfree_extensions import slq as slq_extensions


def problem_setup(suitesparse_matrix_name):
    M = exp_util.suite_sparse_load(suitesparse_matrix_name)

    @jax.jit
    def matvec_(x, p):
        P = jax.experimental.sparse.BCOO((p, M.indices), shape=M.shape)
        return P @ x  # ) + x

    @jax.value_and_grad
    def logdet(p):
        P = jax.experimental.sparse.BCOO((p, M.indices), shape=M.shape).todense()
        # eye = jnp.eye(len(P))
        return jnp.linalg.slogdet(P)[1]

    params = M.data
    #
    # import matplotlib.pyplot as plt
    #
    eigenvalues = jnp.linalg.eigvalsh(M.todense())
    # # print(eigvals)
    # print(sorted(list(set([float(e) for e in eigvals]))))
    # assert False
    # print(set(list(eigvals)))
    #
    #
    # assert False

    value_and_grad_true = logdet(params)
    error_fun = rmse_relative(value_and_grad_true)
    return (matvec_, params), error_fun, M.shape[0], eigenvalues


def rmse_relative(reference, atol=1e-5):
    def error(x, ref):
        absolute = jnp.abs(x - ref)
        normalize = jnp.sqrt(ref.size) * (atol + jnp.abs(ref))
        return jnp.linalg.norm(absolute / normalize)

    return lambda a: jax.tree_util.tree_map(error, a, reference)


def workprecision_avg(keys, *args, **kwargs):
    wps = [workprecision(k, *args, **kwargs) for k in tqdm.tqdm(keys)]
    outer = jax.tree_util.tree_structure(list(keys))
    inner = jax.tree_util.tree_structure(wps[0])
    return jax.tree_util.tree_transpose(outer, inner, wps)


def workprecision(key, orders_, integrand_func, *args, **kwargs):
    wps = []
    for order in tqdm.tqdm(orders_):
        integrand = integrand_func(order)
        wp = workprecision_single(key, integrand, *args, **kwargs)
        wps.append(wp)

        # Gather information for transposing later
        treedef_inner = jax.tree_util.tree_structure(wp)

    # Transpose Pytree to get WP(list()) instead of list(WP())
    treedef_outer = jax.tree_util.tree_structure(list(orders_))
    return jax.tree_util.tree_transpose(treedef_outer, treedef_inner, wps)


def workprecision_single(key, integrand_func, *args, nsamples, nrows, nreps, nbatches):
    # Construct the estimator
    estimate = estimator(
        integrand_func, nsamples=nsamples, nrows=nrows, nbatches=nbatches
    )

    # Estimate and compute error. Precompiles.
    # with jax.disable_jit():
    fx, grad = estimate(key, *args)
    error = error_func((fx, grad))
    # print()
    # print(error)
    # print()
    # assert False
    # Run a bunch of times to evaluate the runtime
    t0 = time.perf_counter()
    for _ in range(nreps):
        value, grad = estimate(key, *args)
        value.block_until_ready()
        grad.block_until_ready()
    t1 = time.perf_counter()

    # Return work-precision information
    return {"error": error, "wall_time": (t1 - t0) / nreps}


def estimator(integrand_func, *, nrows, nsamples, nbatches):
    # integrand_func = jax.value_and_grad(integrand_func, allow_int=True, argnums=1)
    x_like = jnp.ones((nrows,))
    sampler = hutchinson.sampler_rademacher(x_like, num=nsamples)
    estimate_approximate = slq_extensions.hutchinson_nograd(integrand_func, sampler)
    estimate_approximate = slq_extensions.hutchinson_batch(
        estimate_approximate, num=nbatches
    )
    estimate_approximate = jax.value_and_grad(estimate_approximate, argnums=1)
    return jax.jit(estimate_approximate)


if __name__ == "__main__":
    # Set parameters
    num_batches, num_samples_per_batch, num_reps, num_seeds = 10, 10, 1, 1
    # step = (50 - 2) // 4
    orders = [i for i in range(1, 24)]
    print(orders)

    # Set a random key
    prng_key = jax.random.PRNGKey(seed=1)
    key_problem, key_estimate = jax.random.split(prng_key, num=2)

    # Construct a problem
    # bcsstm08:
    # 7 eigvals > 1000, a lot at 1000, another 4 in the low hundreds,
    # another 4 in the tens,
    # and a lot of small ones.
    (matvec, parameters), error_func, num_rows, eigvals = problem_setup("bcsstm08")

    # Run a work precision diagram
    key_estimate_all = jax.random.split(key_estimate, num=num_seeds)

    wps_custom = workprecision_avg(
        key_estimate_all,
        orders,
        lambda o: slq_extensions.integrand_slq_spd_custom_vjp(jnp.log, o, matvec),
        parameters,
        nsamples=num_samples_per_batch,
        nrows=num_rows,
        nreps=num_reps,
        nbatches=num_batches,
    )
    print()
    wps_ref = workprecision_avg(
        key_estimate_all,
        orders[:13],  # memory errors dictate small orders only
        lambda o: slq_extensions.integrand_slq_spd(jnp.log, o, matvec),
        parameters,
        nsamples=num_samples_per_batch,
        nrows=num_rows,
        nreps=num_reps,
        nbatches=num_batches,
    )

    directory = exp_util.matching_directory(__file__, "data/")
    os.makedirs(directory, exist_ok=True)
    jnp.save(f"{directory}/eigvals.npy", eigvals)
    jnp.save(f"{directory}/custom_vjp.npy", wps_custom, allow_pickle=True)
    jnp.save(f"{directory}/reference.npy", wps_ref, allow_pickle=True)
