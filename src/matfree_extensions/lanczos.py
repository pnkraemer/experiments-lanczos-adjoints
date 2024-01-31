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


def tridiag_no_reortho(matvec, krylov_depth, /, *, custom_vjp):
    def estimate(vec, *params):
        return forward_no_reortho(matvec, krylov_depth, vec, *params)

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (jnp.linalg.norm(vec), *params))

    # todo: for full-rank decompositions, the final b_K is almost zero
    #  which blows up the initial step of the backward pass already.
    def estimate_bwd(cache, vjp_incoming):
        # Read incoming gradients and stack related quantities
        (dxs, (dalphas, dbetas)), (dx_last, dbeta_last) = vjp_incoming
        dxs = jnp.concatenate((dxs, dx_last[None]))
        dbetas = jnp.concatenate((dbetas, dbeta_last[None]))

        # Read the cache and stack related quantities
        ((xs, (alphas, betas)), (x_last, beta_last)), (vector_norm, *params) = cache
        xs = jnp.concatenate((xs, x_last[None]))
        betas = jnp.concatenate((betas, beta_last[None]))

        # Compute the adjoints, discard the adjoint states, and return the gradients
        grads, _lambdas = adjoint(
            matvec=matvec,
            params=params,
            initvec_norm=vector_norm,
            alphas=alphas,
            betas=betas,
            xs=xs,
            dalphas=dalphas,
            dbetas=dbetas,
            dxs=dxs,
        )
        return grads

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


def tridiag(matvec, krylov_depth, /, *, custom_vjp):
    def estimate(vec, *params):
        return forward(matvec, krylov_depth, vec, *params)

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (jnp.linalg.norm(vec), *params))

    # todo: for full-rank decompositions, the final b_K is almost zero
    #  which blows up the initial step of the backward pass already.
    def estimate_bwd(cache, vjp_incoming):
        # Read incoming gradients and stack related quantities
        (dxs, (dalphas, dbetas)), (dx_last, dbeta_last) = vjp_incoming
        dxs = jnp.concatenate((dxs, dx_last[None]))
        dbetas = jnp.concatenate((dbetas, dbeta_last[None]))

        # Read the cache and stack related quantities
        ((xs, (alphas, betas)), (x_last, beta_last)), (vector_norm, *params) = cache
        xs = jnp.concatenate((xs, x_last[None]))
        betas = jnp.concatenate((betas, beta_last[None]))

        # Compute the adjoints, discard the adjoint states, and return the gradients
        grads, _lambdas = adjoint(
            matvec=matvec,
            params=params,
            initvec_norm=vector_norm,
            alphas=alphas,
            betas=betas,
            xs=xs,
            dalphas=dalphas,
            dbetas=dbetas,
            dxs=dxs,
        )
        return grads

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


def forward_no_reortho(matvec, krylov_depth, vec, *params):
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
        body_fun=functools.partial(_fwd_step_no_reortho, matvec, params),
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


def _fwd_step_no_reortho(matvec, params, i, val):
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = _fwd_step_apply(matvec, v1, offdiag, v0, *params), v1

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


def adjoint(*, matvec, params, initvec_norm, alphas, betas, xs, dalphas, dbetas, dxs):
    def adjoint_step(xi_and_lambda, inputs):
        return _adjoint_step(*xi_and_lambda, matvec=matvec, params=params, **inputs)

    # Scan over all input gradients and output values
    loop_over = {
        "dx": dxs[:-1],
        "da": dalphas,
        "db": dbetas,
        "xs": (xs[1:], xs[:-1]),
        "a": alphas,
        "b": betas,
    }
    init_val = (-dxs[-1], jnp.zeros_like(dxs[-1]))
    (lambda_1, lambdas), grad_summands = jax.lax.scan(
        adjoint_step,
        init=init_val,
        xs=loop_over,
        reverse=True,
    )

    # Compute the gradients
    grad_matvec = jax.tree_util.tree_map(lambda s: jnp.sum(s, axis=0), grad_summands)
    grad_initvec = ((lambda_1.T @ xs[0]) * xs[0] - lambda_1) / initvec_norm

    # Return values
    return (grad_initvec, grad_matvec), (lambda_1, lambdas)


def _adjoint_step(xi, lambda_plus, /, *, matvec, params, dx, da, db, xs, a, b):
    # Read inputs
    (xplus, x) = xs

    # Apply formula
    xi /= b
    mu = db - lambda_plus.T @ x + xplus.T @ xi
    nu = da + x.T @ xi
    lambda_ = -xi + mu * xplus + nu * x

    # Value-and-grad of matrix-vector product
    matvec_lambda, vjp = jax.vjp(lambda *p: matvec(lambda_, *p), *params)
    (gradient_increment,) = vjp(x)

    # Prepare next step
    xi = -dx - matvec_lambda + a * lambda_ + b * lambda_plus - b * nu * xplus

    # Return values
    return (xi, lambda_), gradient_increment
