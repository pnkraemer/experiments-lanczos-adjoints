"""Extensions for the Matfree package."""


import jax
import jax.flatten_util
import jax.numpy as jnp
from matfree import lanczos


def integrand_spd_custom_vjp(matfun, order, matvec, /):
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

        algorithm = lanczos.alg_tridiag_full_reortho(matvec_flat, order)
        basis, (diag, off_diag) = algorithm(v0_flat, *parameters)

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


def tridiag(matvec, krylov_depth, /):
    def estimate(vec, *params):
        # Pre-allocate
        vectors = jnp.zeros((krylov_depth + 1, len(vec)))
        offdiags = jnp.zeros((krylov_depth,))
        diags = jnp.zeros((krylov_depth,))

        # Normalize (not all Lanczos implementations do that)
        v0 = vec / jnp.linalg.norm(vec)
        vectors = vectors.at[0].set(v0)

        # Lanczos initialisation
        ((v1, offdiag), diag) = _fwd_init(v0, *params)

        # Store results
        k = 0
        vectors = vectors.at[k + 1].set(v1)
        offdiags = offdiags.at[k].set(offdiag)
        diags = diags.at[k].set(diag)

        for k in range(1, krylov_depth):
            # Lanczos step
            ((v1, offdiag), diag), v0 = _fwd_step(v1, offdiag, v0, *params), v1

            # Reorthogonalisation
            v1 = v1 - vectors.T @ (vectors @ v1)
            v1 /= jnp.linalg.norm(v1)

            # Store results
            vectors = vectors.at[k + 1].set(v1)
            offdiags = offdiags.at[k].set(offdiag)
            diags = diags.at[k].set(diag)

        decomposition = vectors[:-1], (diags, offdiags[:-1])
        remainder = vectors[-1], offdiags[-1]
        return decomposition, remainder

    def _fwd_init(vec, *params):
        """Initialize Lanczos' algorithm.

        Solve A x_{k} = a_k x_k + b_k x_{k+1}
        for x_{k+1}, a_k, and b_k, using
        orthogonality of the x_k.
        """
        a = vec @ (matvec(vec, *params))
        r = (matvec(vec, *params)) - a * vec
        b = jnp.linalg.norm(r)
        x = r / b
        return (x, b), a

    def _fwd_step(vec, b, vec_previous, *params):
        """Apply Lanczos' recurrence.

        Solve
        A x_{k} = b_{k-1} x_{k-1} + a_k x_k + b_k x_{k+1}
        for x_{k+1}, a_k, and b_k, using
        orthogonality of the x_k.
        """
        a = vec @ (matvec(vec, *params))
        r = matvec(vec, *params) - a * vec - b * vec_previous
        b = jnp.linalg.norm(r)
        x = r / b
        return (x, b), a

    return estimate
