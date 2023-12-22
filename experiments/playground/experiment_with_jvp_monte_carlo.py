"""Experiment with AD'ing Monte-Carlo methods."""
import jax
import jax.numpy as jnp
from matfree import hutchinson as matfree_hutchinson


def hutchinson(integrand_fun, /, sample_fun, stats_fun=jnp.mean):
    def sample(key, *parameters):
        samples = sample_fun(key)
        Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
        return jax.tree_util.tree_map(lambda s: stats_fun(s, axis=0), Qs)

    return sample


def f(x):
    return x**2


sampler = matfree_hutchinson.sampler_normal(2.0, num=100)

estimate = hutchinson(f, sampler)

print(estimate(jax.random.PRNGKey(1)))
