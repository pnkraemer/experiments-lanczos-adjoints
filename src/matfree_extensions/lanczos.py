"""Extensions for the Matfree package."""

import functools
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp

from matfree_extensions import arnoldi


def integrand_spd(
    matfun: Callable,
    krylov_depth: int,
    matvec: Callable,
    /,
    *,
    reortho: str = "full",
    use_adjoints_for_tridiag: bool = True,
) -> Callable:
    def quadform(v0, *parameters):
        v0_flat, v_unflatten = jax.flatten_util.ravel_pytree(v0)
        scale = jnp.linalg.norm(v0_flat)
        v0_flat /= scale

        @jax.tree_util.Partial
        def matvec_flat(v_flat, *p):
            v = v_unflatten(v_flat)
            av = matvec(v, *p)
            flat, unflatten = jax.flatten_util.ravel_pytree(av)
            return flat

        # We use the efficient VJP for tri-diagonalisation, which implies that this
        # function will be efficiently differentiable
        algorithm = tridiag(
            matvec_flat,
            krylov_depth,
            custom_vjp=use_adjoints_for_tridiag,
            reortho=reortho,
        )
        (basis, (diag, off_diag)), _remainder = algorithm(v0_flat, *parameters)

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
        return scale**2 * jnp.dot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

    return quadform


def integrand_spd_custom_vjp_reuse(matfun, order, matvec, /, *, reortho: str = "full"):
    """Construct an integrand for SLQ for SPD matrices that comes with a custom VJP.

    The custom VJP efficiently computes a single backward-pass (by reusing
    the Lanczos decomposition from the forward pass), but does not admit
    higher derivatives.
    """

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

        # We define our own custom vjp, so no need to select the one for tridiag()
        algorithm = tridiag(matvec_flat, order, custom_vjp=False, reortho=reortho)
        (basis, (diag, off_diag)), _remainder = algorithm(v0_flat, *parameters)

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

    quadform = jax.custom_vjp(quadform)
    quadform.defvjp(quadform_fwd, quadform_bwd)  # type: ignore

    return quadform


def tridiag(matvec, krylov_depth, /, *, reortho: str, custom_vjp: bool = True):
    if reortho == "full":
        return _tridiag_reortho_full(matvec, krylov_depth, custom_vjp=custom_vjp)
    if reortho == "none":
        return _tridiag_reortho_none(matvec, krylov_depth, custom_vjp=custom_vjp)
    raise ValueError


def _tridiag_reortho_full(matvec, krylov_depth, /, *, custom_vjp):
    # Implement via Arnoldi to use the reorthogonalised adjoints.
    # Todo: implement a dedicated function.
    alg = arnoldi.hessenberg(
        matvec, krylov_depth, custom_vjp=custom_vjp, reortho="full"
    )

    def estimate(vec, *params):
        Q, H, v, _norm = alg(vec, *params)

        T = 0.5 * (H + H.T)
        diags = jnp.diag(T, k=0)
        offdiags = jnp.diag(T, k=1)
        decomposition = (Q.T, (diags, offdiags))
        remainder = (v / jnp.linalg.norm(v), jnp.linalg.norm(v))
        return decomposition, remainder

    return estimate


def _tridiag_reortho_none(matvec, krylov_depth, /, *, custom_vjp):
    def estimate(vec, *params):
        *values, _ = _forward(matvec, krylov_depth, vec, *params)
        return values

    def estimate_fwd(vec, *params):
        value = estimate(vec, *params)
        return value, (value, (jnp.linalg.norm(vec), *params))

    # todo: for full-rank decompositions, the final b_K is almost zero
    #  which blows up the initial step of the backward pass already. Solve this!
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
        grads, _lambdas_and_mus_and_nus = _adjoint(
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


def _forward(matvec, krylov_depth, vec, *params):
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
    step_fun = functools.partial(_fwd_step, matvec, params)
    _, (vectors, diags, offdiags) = jax.lax.fori_loop(
        lower=1, upper=krylov_depth, body_fun=step_fun, init_val=init
    )

    # Reorganise the outputs
    decomposition = vectors[:-1], (diags, offdiags[:-1])
    remainder = vectors[-1], offdiags[-1]
    return decomposition, remainder, 1 / jnp.linalg.norm(vec)


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


def _adjoint(*, matvec, params, initvec_norm, alphas, betas, xs, dalphas, dbetas, dxs):
    def adjoint_step(xi_and_lambda, inputs):
        return _adjoint_step(*xi_and_lambda, matvec=matvec, params=params, **inputs)

    # Scan over all input gradients and output values
    xs0 = xs
    xs0 = xs0.at[-1, :].set(jnp.zeros_like(xs[-1, :]))

    loop_over = {
        "dx": dxs[:-1],
        "da": dalphas,
        "db": dbetas,
        "xs": (xs[1:], xs[:-1]),
        "a": alphas,
        "b": betas,
    }
    init_val = (xs0, -dxs[-1], jnp.zeros_like(dxs[-1]))
    (_, lambda_1, _lambda_2), (grad_summands, *other) = jax.lax.scan(
        adjoint_step, init=init_val, xs=loop_over, reverse=True
    )

    # Compute the gradients
    grad_matvec = jax.tree_util.tree_map(lambda s: jnp.sum(s, axis=0), grad_summands)
    grad_initvec = ((lambda_1.T @ xs[0]) * xs[0] - lambda_1) / initvec_norm

    # Return values
    return (grad_initvec, grad_matvec), (lambda_1, *other)


def _adjoint_step(xs_all, xi, lambda_plus, /, *, matvec, params, dx, da, db, xs, a, b):
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
    return (xs_all, xi, lambda_), (gradient_increment, lambda_, mu, nu, xi)
