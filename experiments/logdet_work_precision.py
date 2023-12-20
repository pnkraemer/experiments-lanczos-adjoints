import time
from collections import namedtuple

import jax
import jax.experimental.sparse
import jax.numpy as jnp
from matfree import hutchinson, slq

from matfree_extensions import integrand_slq_spd_custom_vjp


def workprecision(integrand_func, *args, nsamples, nrows, nreps):
    # A utility type
    WP = namedtuple("WP", ["error", "wall_time"])

    # Construct the estimator
    estimate = estimator(integrand_func, nsamples=nsamples, nrows=nrows)

    # Estimate and compute error. Precompiles.
    fx, grad = estimate(*args)
    error = error_func((fx, grad))

    # Run a bunch of times to evaluate the runtime
    t0 = time.perf_counter()
    for _ in range(nreps):
        value, grad = estimate(*args)
        value.block_until_ready()
        grad.block_until_ready()
    t1 = time.perf_counter()

    # Return work-precision information
    return WP(error, (t1 - t0) / nreps)


def estimator(integrand_func, *, nrows, nsamples):
    integrand_func = jax.value_and_grad(integrand_func, allow_int=True, argnums=1)
    x_like = jnp.ones((nrows,))
    sampler = hutchinson.sampler_rademacher(x_like, num=nsamples)
    estimate_approximate = hutchinson.hutchinson(integrand_func, sampler)
    estimate_approximate = jax.jit(estimate_approximate)
    return estimate_approximate


def rmse_relative(reference, atol=1):
    def error(x, ref):
        absolute = jnp.abs(x - ref)
        normalize = jnp.sqrt(ref.size) * jnp.abs(atol + ref)
        return jnp.linalg.norm(absolute / normalize)

    return lambda a: jax.tree_util.tree_map(error, a, reference)


def problem_setup(key, *, nrows):
    def matvec(x, p):
        return p * x

    @jax.value_and_grad
    def logdet(p):
        return jnp.sum(jnp.log(p))

    parameters = 1 + jax.random.uniform(key, shape=(nrows,))
    value_and_grad_true = logdet(parameters)
    error_fun = rmse_relative(value_and_grad_true)
    return (matvec, parameters), error_fun


if __name__ == "__main__":
    n = 100
    num_samples = 1
    seed = 1
    num_reps = 1

    prng_key = jax.random.PRNGKey(seed)
    key_parameter, key_estimate = jax.random.split(prng_key, num=2)

    (matvec, parameters), error_func = problem_setup(key_parameter, nrows=n)

    for order in range(1, 10):
        # integrand = slq.integrand_slq_spd(jnp.log, order, matvec)
        integrand = integrand_slq_spd_custom_vjp(jnp.log, order, matvec)
        wp = workprecision(
            integrand,
            key_estimate,
            parameters,
            nsamples=num_samples,
            nrows=n,
            nreps=num_reps,
        )
        print(wp)
