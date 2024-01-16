"""Extensions for the Matfree package."""

import jax
import jax.flatten_util
import jax.numpy as jnp


def hutchinson_nograd(integrand_fun, /, sample_fun):
    """Implement Hutchinson's estimator but stop the gradients through the samples."""

    def sample(key, *parameters):
        samples = sample_fun(key)
        samples = jax.lax.stop_gradient(samples)
        Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)

    return jax.jit(sample)


def hutchinson_custom_vjp(integrand_fun, /, sample_fun):
    """Implement Hutchinson's estimator but use a different key during the backward pass."""

    @jax.custom_vjp
    def sample(_key, *_parameters):
        #
        # This function shall only be meaningful inside a VJP,
        # thus, we raise a:
        #
        raise RuntimeError("oops")

    def sample_fwd(key, *parameters):
        _key_fwd, key_bwd = jax.random.split(key, num=2)
        sampled = _sample(sample_fun, integrand_fun, key, *parameters)
        return sampled, {"key": key_bwd, "parameters": parameters}

    def sample_bwd(cache, vjp_incoming):
        def integrand_fun_new(v, *p):
            # this is basically a checkpoint?
            _fx, vjp = jax.vjp(integrand_fun, v, *p)
            return vjp(vjp_incoming)

        key = cache["key"]
        parameters = cache["parameters"]
        return _sample(sample_fun, integrand_fun_new, key, *parameters)

    sample.defvjp(sample_fwd, sample_bwd)
    return sample


def _sample(sample_fun, integrand_fun, key, *parameters):
    samples = sample_fun(key)
    Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
    return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)


def hutchinson_batch(estimate_fun, /, num):
    """Batched-call the results of Hutchinson's estimator."""

    def estimate_b(key, *parameters):
        keys = jax.random.split(key, num=num)
        estimates = jax.lax.map(lambda k: estimate_fun(k, *parameters), keys)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), estimates)

    return jax.jit(estimate_b)
