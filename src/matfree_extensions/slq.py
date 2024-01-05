"""Extensions for the Matfree package."""

import jax
import jax.flatten_util
import jax.numpy as jnp
from matfree import decomp


def integrand_slq_spd_value_and_grad(matfun, order, matvec, /):
    def quadform(v0, *parameters):
        v0_flat, v_unflatten = jax.flatten_util.ravel_pytree(v0)
        scale = jnp.linalg.norm(v0_flat)
        v0_flat /= scale

        def matvec_flat(v_flat):
            v = v_unflatten(v_flat)
            Av = matvec(v, *parameters)
            flat, unflatten = jax.flatten_util.ravel_pytree(Av)
            return flat

        algorithm = decomp.lanczos_tridiag_full_reortho(order)
        basis, tridiag = decomp.decompose_fori_loop(
            v0_flat, matvec_flat, algorithm=algorithm
        )
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = jnp.diag(diag)
        offdiag1 = jnp.diag(off_diag, -1)
        offdiag2 = jnp.diag(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = jnp.linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0_flat.shape

        # Evaluate the matrix-function
        fx_eigvals = jax.vmap(matfun)(eigvals)
        slqval = dim * jnp.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

        # Evaluate the derivative
        dfx_eigvals = jax.vmap(jax.jacfwd(matfun))(eigvals)
        sol = eigvecs @ (dfx_eigvals * eigvecs[0, :].T)
        w1, w2 = jnp.linalg.norm(v0) * (basis.T @ sol), v0

        @jax.tree_util.Partial
        def matvec_flat_p(v_flat, *p):
            v = v_unflatten(v_flat)
            av = matvec(v, *p)
            flat, unflatten = jax.flatten_util.ravel_pytree(av)
            return flat

        grad = jax.grad(lambda *pa: matvec_flat_p(w2, *pa).T @ w1)(*parameters)

        return slqval, grad

    return quadform


def integrand_slq_spd_custom_vjp(matfun, order, matvec, /):
    @jax.custom_vjp
    def quadform(_v0, *_parameters):
        #
        # This function shall only be meaningful inside a VJP,
        # thus, we raise a:
        #
        raise RuntimeError

    def quadform_fwd(v0, *parameters):
        v0_flat, v_unflatten = jax.flatten_util.ravel_pytree(v0)
        scale = jnp.linalg.norm(v0_flat)
        v0_flat /= scale

        @jax.tree_util.Partial
        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            av = matvec(v, *p)
            flat, unflatten = jax.flatten_util.ravel_pytree(av)
            return flat

        algorithm = decomp.lanczos_tridiag_full_reortho(order)
        basis, tridiag = decomp.decompose_fori_loop(
            v0_flat, lambda v: matvec_flat(v, *parameters), algorithm=algorithm
        )
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = jnp.diag(diag)
        offdiag1 = jnp.diag(off_diag, -1)
        offdiag2 = jnp.diag(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = jnp.linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0_flat.shape

        # Evaluate the matrix-function
        fx_eigvals = jax.vmap(matfun)(eigvals)
        slqval = dim * jnp.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

        # Evaluate the derivative
        dfx_eigvals = jax.vmap(jax.jacfwd(matfun))(eigvals)
        sol = eigvecs @ (dfx_eigvals * eigvecs[0, :].T)
        w1, w2 = dim * (basis.T @ sol), v0_flat

        # Return both
        cache = {
            "matvec_flat": matvec_flat,
            "w1": w1,
            "w2": w2,
            "parameters": parameters,
            "dim": dim,
        }
        return slqval, cache

    def quadform_bwd(cache, vjp_incoming):
        matvec_flat = cache["matvec_flat"]
        p = cache["parameters"]
        w1, w2 = cache["w1"], cache["w2"]
        d = cache["dim"]

        fx, vjp = jax.vjp(lambda *pa: 1 / d * matvec_flat(w2, *pa).T @ w1, *p)
        # todo: compute gradient wrt v?
        return 0.0, *vjp(d * vjp_incoming)

    quadform.defvjp(quadform_fwd, quadform_bwd)

    return quadform


def hutchinson_nograd(integrand_fun, /, sample_fun):
    def sample(key, *parameters):
        samples = sample_fun(key)
        samples = jax.lax.stop_gradient(samples)
        Qs = jax.vmap(lambda vec: integrand_fun(vec, *parameters))(samples)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), Qs)

    return jax.jit(sample)


def hutchinson_custom_vjp(integrand_fun, /, sample_fun):
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
    def estimate_b(key, *parameters):
        keys = jax.random.split(key, num=num)
        estimates = jax.lax.map(lambda k: estimate_fun(k, *parameters), keys)
        return jax.tree_util.tree_map(lambda s: jnp.mean(s, axis=0), estimates)

    return jax.jit(estimate_b)


def integrand_slq_spd(matfun, order, matvec, /):
    """Quadratic form for stochastic Lanczos quadrature.

    This function assumes a symmetric, positive definite matrix.
    """

    def quadform(v0, *parameters):
        v0_flat, v_unflatten = jax.flatten_util.ravel_pytree(v0)

        def matvec_flat(v_flat):
            v = v_unflatten(v_flat)
            Av = matvec(v, *parameters)
            flat, unflatten = jax.flatten_util.ravel_pytree(Av)
            return flat

        algorithm = decomp.lanczos_tridiag_full_reortho(order)
        _, tridiag = decomp.decompose_fori_loop(
            v0_flat, matvec_flat, algorithm=algorithm
        )
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = jnp.diag(diag)
        offdiag1 = jnp.diag(off_diag, -1)
        offdiag2 = jnp.diag(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = jnp.linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0_flat.shape

        fx_eigvals = jax.vmap(matfun)(eigvals)
        return dim * jnp.linalg.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform
