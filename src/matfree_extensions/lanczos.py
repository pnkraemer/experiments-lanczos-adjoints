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
        return forward(matvec, krylov_depth, vec, *params)

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (vec, *params))

    # todo: for full-rank decompositions, the final b_K is almost zero
    #  which blows up the initial step of the backward pass already.
    def estimate_bwd(cache, vjp_incoming):
        # Read incoming gradients and stack related quantities
        (dxs, (dalphas, dbetas)), (dx_last, dbeta_last) = vjp_incoming
        dxs = jnp.concatenate((dxs, dx_last[None]))
        dbetas = jnp.concatenate((dbetas, dbeta_last[None]))

        # Read the cache and stack related quantities
        ((xs, (alphas, betas)), (x_last, beta_last)), (vector, *params) = cache
        xs = jnp.concatenate((xs, x_last[None]))
        betas = jnp.concatenate((betas, beta_last[None]))

        return adjoint(
            matvec,
            alphas=alphas,
            betas=betas,
            dalphas=dalphas,
            dbetas=dbetas,
            dxs=dxs,
            params=params,
            vector=vector,
            xs=xs,
        )

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


def forward(matvec, krylov_depth, vec, *params):
    # Pre-allocate
    vectors = jnp.zeros((krylov_depth + 1, len(vec)))
    offdiags = jnp.zeros((krylov_depth,))
    diags = jnp.zeros((krylov_depth,))

    # Normalize (not all Lanczos implementations do that)
    v0 = vec / jnp.linalg.norm(vec)
    vectors = vectors.at[0].set(v0)

    # Lanczos initialisation
    ((v1, offdiag), diag) = _fwd_init(matvec, v0, *params)

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
        body_fun=functools.partial(_fwd_step, matvec, params),
        init_val=init,
    )

    # Reorganise the outputs
    decomposition = vectors[:-1], (diags, offdiags[:-1])
    remainder = vectors[-1], offdiags[-1]
    return decomposition, remainder


def _fwd_init(matvec, vec, *params):
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


def _fwd_step(matvec, params, i, val):
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = _fwd_step_apply(matvec, v1, offdiag, v0, *params), v1

    # Reorthogonalisation
    v1 = v1 - vectors.T @ (vectors @ v1)
    v1 /= jnp.linalg.norm(v1)

    # Store results
    vectors = vectors.at[i + 1].set(v1)
    offdiags = offdiags.at[i].set(offdiag)
    diags = diags.at[i].set(diag)

    return (v1, offdiag, v0), (vectors, diags, offdiags)


def _fwd_step_apply(matvec, vec, b, vec_previous, *params):
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


def adjoint(matvec, *, params, vector, alphas, betas, xs, dalphas, dbetas, dxs):
    # Initialise the states
    (xi, lambda_k), dA_increment = _bwd_init(
        matvec=matvec,
        params=params,
        da=dalphas[-1],
        db=dbetas[-1],
        a=alphas[-1],
        b=betas[-1],
        dxs=(dxs[-1], dxs[-2]),
        xs=(xs[-1], xs[-2]),
    )

    # Scan over the remaining inputs
    loop_over = (
        (dxs[:-2]),
        dalphas[:-1],
        dbetas[:-1],
        (xs[1:-1], xs[:-2]),
        alphas[:-1],
        (betas[:-1]),
    )
    init_val = (xi, lambda_k)

    def body_fun(carry, x):
        xi_, lambda_k_ = carry
        output_ = _bwd_step(matvec, xi_, lambda_k_, inputs=x, params=params)
        (xi_, lambda_k_), da_increment = output_
        return (xi_, lambda_k_), da_increment

    (xi, lambdas), dAs = jax.lax.scan(
        body_fun,
        init=init_val,
        xs=loop_over,
        reverse=True,
    )
    dA = jnp.sum(dAs, axis=0) + dA_increment

    # Conclude the final step:
    lambda_1 = -xi  # / betas[0]
    # # todo: also return all lambdas

    # Compute the gradients
    dv = ((lambda_1.T @ xs[0]) * xs[0] - lambda_1) / jnp.linalg.norm(vector)
    return dv, dA


def _bwd_init(*, matvec, params, a, b, xs, da, db, dxs):
    # Read inputs
    xplus, x = xs
    dxplus, dx = dxs

    # Apply formula
    xi = dxplus / b
    mu_K = db - xplus.T @ xi
    nu_K = da - x.T @ xi
    lambda_Kplus = xi + mu_K * xplus + nu_K * x

    #
    Av, vjp = jax.vjp(lambda *p: matvec(lambda_Kplus, *p), *params)
    (dA_increment,) = vjp(x)

    # Apply formula
    lambda_Kplusplus = jnp.zeros_like(lambda_Kplus)
    xi = dx + Av - a * lambda_Kplus - b * lambda_Kplusplus + b * nu_K * xplus
    return (xi, lambda_Kplus), dA_increment


def _bwd_step(matvec, xi, lambda_kplus, /, *, inputs, params):
    dx_Kminus, da_Kminus, db_Kminus, (x_K, x_Kminus), a_Kminus, b_Kminus = inputs

    # Apply formula
    xi /= b_Kminus
    mu_Kminus = db_Kminus - lambda_kplus.T @ x_Kminus - x_K.T @ xi
    nu_Kminus = da_Kminus - x_Kminus.T @ xi
    lambda_K = xi + mu_Kminus * x_K + nu_Kminus * x_Kminus

    # Prepare next step
    Av, vjp = jax.vjp(lambda *p: matvec(lambda_K, *p), *params)
    (dA_increment,) = vjp(x_Kminus)
    xi = (
        dx_Kminus
        + Av
        - a_Kminus * lambda_K
        - b_Kminus * lambda_kplus
        + b_Kminus * nu_Kminus * x_K
    )

    return (xi, lambda_K), dA_increment
