"""Extensions for the Matfree package."""

import functools

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


def tridiag(matvec, krylov_depth, /, *, custom_vjp):
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

        # Run Lanczos-loop
        init = (v1, offdiag, v0), (vectors, diags, offdiags)
        _, (vectors, diags, offdiags) = jax.lax.fori_loop(
            lower=1,
            upper=krylov_depth,
            body_fun=functools.partial(_fwd_step, params),
            init_val=init,
        )

        # Reorganise the outputs
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

    def _fwd_step(params, i, val):
        (v1, offdiag, v0), (vectors, diags, offdiags) = val
        ((v1, offdiag), diag), v0 = _fwd_step_apply(v1, offdiag, v0, *params), v1

        # Reorthogonalisation
        v1 = v1 - vectors.T @ (vectors @ v1)
        v1 /= jnp.linalg.norm(v1)

        # Store results
        vectors = vectors.at[i + 1].set(v1)
        offdiags = offdiags.at[i].set(offdiag)
        diags = diags.at[i].set(diag)

        return (v1, offdiag, v0), (vectors, diags, offdiags)

    def _fwd_step_apply(vec, b, vec_previous, *params):
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

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (vec, *params))

    def estimate_bwd(cache, vjp_incoming):
        # Read incoming gradients and stack related quantities
        (dxs, (dalphas, dbetas)), (dx_last, dbeta_last) = vjp_incoming
        dxs = jnp.concatenate((dxs, dx_last[None]))
        dbetas = jnp.concatenate((dbetas, dbeta_last[None]))

        # Read the cache and stack related quantities
        ((xs, (alphas, betas)), (x_last, beta_last)), (vector, *params) = cache
        xs = jnp.concatenate((xs, x_last[None]))
        betas = jnp.concatenate((betas, beta_last[None]))

        # Initialise the states
        nu, lambda_k = _bwd_init(
            dx_Kplus=dxs[-1],
            da_K=dalphas[-1],
            db_K=dbetas[-1],
            b_K=betas[-1],
            x_Ks=(xs[-1], xs[-2]),
        )
        zeros = jnp.zeros_like(lambda_k)
        lambdas = (zeros, lambda_k)
        dA = 0.0  # todo: for multiple parameters, this should be a tree_zeros!

        for k in reversed(range(1, len(alphas))):
            nu, lambda_k, dA_increment = _bwd_step(
                *params,
                lambdas=lambdas,
                nu_K=nu,
                dx_K=dxs[k],
                da_Kminus=dalphas[k - 1],
                db_Kminus=dbetas[k - 1],
                a_K=alphas[k],
                b_Ks=(betas[k], betas[k - 1]),
                x_Ks=(xs[k + 1], xs[k], xs[k - 1]),
            )
            lambdas = (lambdas[1], lambda_k)

            # todo: for multiple parameters, this should be a tree_add!
            dA += dA_increment

        lambda_1, dA_increment = _bwd_final(
            *params,
            lambdas=lambdas,
            nu_K=nu,
            b_K=betas[0],
            a_K=alphas[0],
            x_Ks=(xs[1], xs[0]),
            dx_K=dxs[0],
        )

        dA += dA_increment
        dv = ((lambda_1.T @ xs[0]) * xs[0] - lambda_1) / jnp.linalg.norm(vector)
        return dv, dA

    def _bwd_init(*, dx_Kplus, da_K, db_K, b_K, x_Ks):
        # Read inputs
        x_Kplus, x_K = x_Ks

        # Apply formula
        mu_K = db_K * b_K - x_Kplus.T @ dx_Kplus
        nu_K = da_K * b_K - x_K.T @ dx_Kplus
        lambda_Kplus = (dx_Kplus + mu_K * x_Kplus + nu_K * x_K) / b_K
        return nu_K, lambda_Kplus

    def _bwd_step(*params, lambdas, dx_K, db_Kminus, da_Kminus, a_K, b_Ks, nu_K, x_Ks):
        lambda_Kplusplus, lambda_kplus = lambdas
        x_Kplus, x_K, x_Kminus = x_Ks
        b_K, b_Kminus = b_Ks

        Av, vjp = jax.vjp(lambda *p: matvec(lambda_kplus, *p), *params)
        (dA_increment,) = vjp(x_K)

        xi = dx_K + Av - a_K * lambda_kplus - b_K * lambda_Kplusplus + nu_K * x_Kplus
        mu_Kminus = b_Kminus * (db_Kminus - lambda_kplus @ x_Kminus) - x_K.T @ xi
        nu_Kminus = b_Kminus * da_Kminus - x_Kminus.T @ xi
        lambda_K = (xi + mu_Kminus * x_K + nu_Kminus * x_Kminus) / b_Kminus
        return nu_Kminus, lambda_K, dA_increment

    def _bwd_final(*params, lambdas, x_Ks, nu_K, b_K, a_K, dx_K):
        x1, x0 = x_Ks
        lambda_Kplus, lambda_K = lambdas

        Av, vjp = jax.vjp(lambda *p: matvec(lambda_K, *p), *params)
        (dA_increment,) = vjp(x0)

        lambda_1 = b_K * lambda_Kplus - Av + a_K * lambda_K - nu_K * x1 - dx_K
        return lambda_1, dA_increment

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate
