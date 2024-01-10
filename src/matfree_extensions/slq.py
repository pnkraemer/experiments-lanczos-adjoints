"""Extensions for the Matfree package."""
import functools

import jax
import jax.flatten_util
import jax.numpy as jnp
from matfree import decomp


def integrand_slq_spd_value_and_grad(matfun, order, matvec, /):
    """Construct an integrand that computes the value and gradient of SLQ for SPD matrices.

    This yields E[value_and_grad()], and is therefore neither forward- nor
    reverse-mode. It is rather a "clever implementation" of what is common
    in the GP community (clever because it requires a single backward-pass
    over a parameter-to-scalar function instead of propagating kernel parameters forward).

    Use this function if SLQ is the entire computational chain, i.e.,
    if neither forward- nor reverse-mode are strictly required.
    """

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
    """Construct an integrand for SLQ for SPD matrices that comes with a custom VJP.

    The custom VJP efficiently computes a single backward-pass (by reusing
    the Lanczos decomposition from the forward pass), but does not admit
    higher derivatives.
    """

    @jax.custom_vjp
    def quadform(v0, *parameters):
        return quadform_fwd(v0, *parameters)[0]
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
        # Evaluate the matrix-function
        fx_eigvals = jax.vmap(matfun)(eigvals)
        slqval = scale**2 * jnp.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

        # Evaluate the derivative
        dfx_eigvals = jax.vmap(jax.jacfwd(matfun))(eigvals)
        sol = eigvecs @ (dfx_eigvals * eigvecs[0, :].T)
        w1, w2 = scale**2 * (basis.T @ sol), v0_flat

        # Return both
        cache = {
            "matvec_flat": matvec_flat,
            "w1": w1,
            "w2": w2,
            "parameters": parameters,
        }
        return slqval, cache

    def quadform_bwd(cache, vjp_incoming):
        matvec_flat = cache["matvec_flat"]
        p = cache["parameters"]
        w1, w2 = cache["w1"], cache["w2"]

        fx, vjp = jax.vjp(lambda *pa: matvec_flat(w2, *pa).T @ w1, *p)

        # todo: compute gradient wrt v?
        return 0.0, *vjp(vjp_incoming)

    quadform.defvjp(quadform_fwd, quadform_bwd)

    return quadform


def integrand_slq_spd(matfun, order, matvec, /):
    """Quadratic form for stochastic Lanczos quadrature.

    This function assumes a symmetric, positive definite matrix.

    This is a mirror of matfree.slq.integrand_slq_spd, but can be modified
    whenever required.
    """

    def quadform(v0, *parameters):
        v0_flat, v_unflatten = jax.flatten_util.ravel_pytree(v0)
        length = jnp.linalg.norm(v0_flat)
        v0_flat /= length

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

        fx_eigvals = jax.vmap(matfun)(eigvals)
        return length**2 * jnp.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def integrand_slq_spd_custom_vjp_recursive(matfun, order, matvec, /):
    """Construct an integrand for SLQ for SPD matrices that comes with a custom VJP.

    The custom VJP recursively calls into quadform(), and as such, allows higher derivatives.
    But this comes at the price of calling Lanczos twice more in the backward pass,
    which makes it more costly for computing gradients.
    """

    def quadform(v0, *parameters):
        return _integrand_slq(matfun, order, matvec, v0, *parameters)

    return quadform


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _integrand_slq(matfun, order, matvec, v0, *parameters):
    return _integrand_slq_fwd(matfun, order, matvec, v0, v0, *parameters)[0]


def _integrand_slq_fwd(matfun, order, matvec, v0, *parameters):
    v0_flat_unscaled, v_unflatten = jax.flatten_util.ravel_pytree(v0)
    scale = jnp.linalg.norm(v0_flat_unscaled)
    v0_flat = v0_flat_unscaled / scale

    def matvec_flat(v_flat, *p):
        v = v_unflatten(v_flat)
        av = matvec(v, *p)
        flat, unflatten = jax.flatten_util.ravel_pytree(av)
        return flat

    # Lanczos decomposition
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

    # Stop gradients through Lanczos and eigenvalues
    basis, tridiag = jax.tree_util.tree_map(jax.lax.stop_gradient, (basis, tridiag))
    eigvals, eigvecs = jax.tree_util.tree_map(jax.lax.stop_gradient, (eigvals, eigvecs))

    # Do not explot that basis @ v0_flat is e1, because that would yield
    # the wrong gradients.
    z = eigvecs.T @ (basis @ v0_flat)

    # Evaluate the matrix-function and the SLQ-quadratic form
    fx_eigvals = jax.vmap(matfun)(eigvals)
    slqval = scale**2 * jnp.dot(z, fx_eigvals * z)

    # Return the SLQ value and cache v0 as well as the parameters.
    cache = {"v0": v_unflatten(v0_flat_unscaled), "parameters": parameters}
    return slqval, cache


def _integrand_slq_bwd(matfun, order, matvec, cache, vjp_incoming):
    parameters = cache["parameters"]
    v0 = cache["v0"]

    def evaluate_asymmetric_quantity(*pa):
        """Evaluate v^\top f(A) (Av) via asymmetric slq."""
        mv = matvec(v0, *pa)
        z1 = v0 + mv  # todo: tree_map
        z2 = v0 - mv  # todo: tree_map

        # These use stop_gradient(lanczos), so differentiation should be almost free.
        Z1, _ = _integrand_slq_fwd(jax.jacrev(matfun), order, matvec, z1, *pa)
        Z2, _ = _integrand_slq_fwd(jax.jacrev(matfun), order, matvec, z2, *pa)
        return (Z1 - Z2) / 4

    # _fx is irrelevant for VJPs, but useful for debugging
    # For full-order approximations _fx == ||v0||^2 and if not, something is wrong.
    _fx, vjp = jax.vjp(evaluate_asymmetric_quantity, *parameters)

    return 0.0, *vjp(vjp_incoming)


_integrand_slq.defvjp(_integrand_slq_fwd, _integrand_slq_bwd)


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
