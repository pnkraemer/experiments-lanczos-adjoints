import time

import jax
import jax.experimental.sparse
import jax.numpy as jnp
import tqdm
from matfree import hutchinson, slq

from matfree_extensions import bcoo_random_spd, integrand_slq_spd_custom_vjp


def problem_setup(key, *, nrows):
    M = bcoo_random_spd(key, num_rows=nrows, num_nonzeros=2 * nrows)
    params, indices = M.data, M.indices

    @jax.jit
    def matvec_(x, p):
        P = jax.experimental.sparse.BCOO((p, indices), shape=(nrows, nrows))
        return P.T @ (P @ x) + x

    @jax.value_and_grad
    def logdet(p):
        P = jax.experimental.sparse.BCOO((p, indices), shape=(nrows, nrows)).todense()
        return jnp.linalg.slogdet(P.T @ P + jnp.eye(nrows))[1]

    #
    # # eigvals = 1 + jnp.arange(0, nrows, step=1, dtype=float)
    # eigvals = 1.0 + jax.random.uniform(key, shape=(nrows,))
    # # eigvals = eigvals.at[-2].set(eigvals[-1])
    # params = test_util.symmetric_matrix_from_eigenvalues(eigvals)

    value_and_grad_true = logdet(params)
    error_fun = rmse_relative(value_and_grad_true)
    return (matvec_, params), error_fun


def rmse_relative(reference, atol=1):
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


def workprecision(key, os, integrand_func, *args, **kwargs):
    wps = []
    for order in tqdm.tqdm(os):
        integrand = integrand_func(order)
        wp = workprecision_single(key, integrand, *args, **kwargs)
        wps.append(wp)

        # Gather information for transposing later
        treedef_inner = jax.tree_util.tree_structure(wp)

    # Transpose Pytree to get WP(list()) instead of list(WP())
    treedef_outer = jax.tree_util.tree_structure(list(os))
    return jax.tree_util.tree_transpose(treedef_outer, treedef_inner, wps)


def workprecision_single(key, integrand_func, *args, nsamples, nrows, nreps):
    # Construct the estimator
    estimate = estimator(integrand_func, nsamples=nsamples, nrows=nrows)

    # Estimate and compute error. Precompiles.
    fx, grad = estimate(key, *args)
    error = error_func((fx, grad))

    # Run a bunch of times to evaluate the runtime
    t0 = time.perf_counter()
    for _ in range(nreps):
        value, grad = estimate(key, *args)
        value.block_until_ready()
        grad.block_until_ready()
    t1 = time.perf_counter()

    # Return work-precision information
    return {"error": error, "wall_time": (t1 - t0) / nreps}


def estimator(integrand_func, *, nrows, nsamples):
    integrand_func = jax.value_and_grad(integrand_func, allow_int=True, argnums=1)
    x_like = jnp.ones((nrows,))
    sampler = hutchinson.sampler_normal(x_like, num=nsamples)
    estimate_approximate = hutchinson.hutchinson(integrand_func, sampler)
    estimate_approximate = jax.jit(estimate_approximate)
    return estimate_approximate


if __name__ == "__main__":
    # Set parameters
    num_rows, num_samples, num_reps, num_seeds = 1000, 1000, 1, 5
    # step = (50 - 2) // 4
    orders = [2**i for i in range(0, 7)]
    print(orders)

    # Set a random key
    prng_key = jax.random.PRNGKey(seed=1)
    key_problem, key_estimate = jax.random.split(prng_key, num=2)

    # Construct a problem
    (matvec, parameters), error_func = problem_setup(key_problem, nrows=num_rows)

    # Run a work precision diagram
    key_estimate_all = jax.random.split(key_estimate, num=num_seeds)
    wps_ref = workprecision_avg(
        key_estimate_all,
        orders[:4],  # memory errors dictate small orders only
        lambda o: slq.integrand_slq_spd(jnp.log, o, matvec),
        parameters,
        nsamples=num_samples,
        nrows=num_rows,
        nreps=num_reps,
    )
    print()
    wps_custom = workprecision_avg(
        key_estimate_all,
        orders,
        lambda o: integrand_slq_spd_custom_vjp(jnp.log, o, matvec),
        parameters,
        nsamples=num_samples,
        nrows=num_rows,
        nreps=num_reps,
    )

    jnp.save("./data/workprecision_dense_custom_vjp.npy", wps_custom, allow_pickle=True)
    jnp.save("./data/workprecision_dense_reference.npy", wps_ref, allow_pickle=True)
