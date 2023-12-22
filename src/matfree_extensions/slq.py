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
        print(fx)
        # todo: compute gradient wrt v?
        return 0.0, *vjp(d * vjp_incoming)

    quadform.defvjp(quadform_fwd, quadform_bwd)

    return quadform
