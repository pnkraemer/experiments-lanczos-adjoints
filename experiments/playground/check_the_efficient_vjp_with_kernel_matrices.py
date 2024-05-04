import functools

import jax
import jax.numpy as jnp
from matfree import hutchinson
from matfree_extensions.util import gp_util


def gram_matvec(kfun, p, x, y, v):
    gram = gp_util.gram_matvec_full_batch()
    return gram(kfun(**p))(x, y, v)


def gram_matvec_cheap(kfun, p, x, y, v):
    gram = gp_util.gram_matvec_full_batch()
    return gram(kfun(**p))(x, y, v)


def gram_matvec_fwd(function, p, x, y, v):
    return gram_matvec(function, p, x, y, v), (p, x, y, v)


def gram_matvec_rev(function, cache, vjp_incoming):
    def reduce(p, x, y, v):
        kx = gram_matvec_cheap(function, p, y, x, vjp_incoming)
        return jnp.dot(v, kx)

    return jax.grad(reduce, argnums=(0, 1, 2, 3))(*cache)


#
gram_matvec = jax.custom_vjp(gram_matvec, nondiff_argnums=[0])
gram_matvec.defvjp(gram_matvec_fwd, gram_matvec_rev)

N = 40_000
input_dim = 1
krylov_depth = 1
vec = jnp.ones((N,))
xs = jnp.linspace(0, 1, num=N)
ys = xs

sampler = hutchinson.sampler_rademacher(vec, num=1)
_logpdf, logdet = gp_util.logpdf_lanczos(
    krylov_depth, sampler, slq_batch_num=1, cg_tol=1.0, checkpoint=False
)

k, p_prior = gp_util.kernel_scaled_rbf(shape_in=(), shape_out=(), checkpoint=False)


@jax.jit
@jax.value_and_grad
def fun(p):
    cov = functools.partial(gram_matvec, k, p, xs, xs)
    return logdet(cov, jax.random.PRNGKey(1))


print(fun(p_prior))
