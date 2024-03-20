"""Experiment with AD'ing Monte-Carlo methods."""

import functools

import jax
import jax.numpy as jnp
from matfree import hutchinson as matfree_hutchinson


def hutchinson_share(integrand_fun, /, sample_fun):
    # todo: what is the derivative wrt the key?
    # todo: do we HAVE to reuse the same monte-Carlo samples?
    @functools.partial(jax.custom_jvp, nondiff_argnums=[0])
    def sample(k, *parameters):
        samples = sample_fun(k)
        Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)

    def sample_jvp(k_primals, p_primals, p_tangents):
        samples = sample_fun(k_primals)

        def integrand_jvp(vec):
            integrand_p = functools.partial(integrand_fun, vec)
            return jax.jvp(integrand_p, p_primals, p_tangents)

        Qs = jax.vmap(integrand_jvp)(samples)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)

    sample.defjvp(sample_jvp)
    return sample


def f(x, p):
    return p * x**2


sampler = matfree_hutchinson.sampler_normal(2.0, num=1)
estimate_share = hutchinson_share(f, sampler)
key = jax.random.PRNGKey(1)
estimated_share = estimate_share(key, 2.0)
print("Primals", estimated_share)
estimated_share, jvp_share = jax.jvp(
    functools.partial(estimate_share, key), [2.0], [1.0]
)
print("Primals (again)", estimated_share)
print("JVP", jvp_share)
print()


estimate_matfree = matfree_hutchinson.hutchinson(f, sampler)
key = jax.random.PRNGKey(1)
estimated_matfree = estimate_matfree(key, 2.0)
print("Primals", estimated_matfree)
estimated_matfree, jvp_matfree = jax.jvp(
    functools.partial(estimate_matfree, key), [2.0], [1.0]
)
print("Primals (again)", estimated_matfree)
print("JVP", jvp_matfree)
print()
