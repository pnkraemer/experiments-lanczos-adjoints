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


def tridiag(matvec, krylov_depth, /, *, custom_vjp, reortho=True):
    def estimate(vec, *params):
        *values, _ = forward(matvec, krylov_depth, vec, *params, reortho=reortho)
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
        grads, _lambdas_and_mus_and_nus = adjoint(
            matvec=matvec,
            params=params,
            initvec_norm=vector_norm,
            alphas=alphas,
            betas=betas,
            xs=xs,
            dalphas=dalphas,
            dbetas=dbetas,
            dxs=dxs,
            # Always set to 'False' until we figure this out properly.
            reortho=False,
        )
        return grads

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


def matrix_forward(matvec, krylov_depth, vec, *params):
    output = forward(matvec, krylov_depth, vec, *params, reortho=True)
    (Q, (alphas, betas)), (q, beta), norm = output
    return Q, (alphas, betas), q * beta, norm


def forward(matvec, krylov_depth, vec, *params, reortho):
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
    step_fun = functools.partial(_fwd_step, matvec, params, reortho=reortho)
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


def _fwd_step(matvec, params, i, val, *, reortho: bool):
    (v1, offdiag, v0), (vectors, diags, offdiags) = val
    ((v1, offdiag), diag), v0 = _fwd_step_apply(matvec, v1, offdiag, v0, *params), v1

    # Reorthogonalisation
    if reortho:
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


def adjoint(
    *, matvec, params, initvec_norm, alphas, betas, xs, dalphas, dbetas, dxs, reortho
):
    if not reortho:
        return _adjoint_pass(
            matvec=matvec,
            params=params,
            initvec_norm=initvec_norm,
            alphas=alphas,
            betas=betas,
            xs=xs,
            dalphas=dalphas,
            dbetas=dbetas,
            dxs=dxs,
            reortho=reortho,
        )

    dxs_in = dxs @ (xs.T @ xs)
    dxs_out = dxs - dxs_in

    dalphas_out = dalphas
    dbetas_out = dbetas
    output_out = _adjoint_pass(
        matvec=matvec,
        params=params,
        initvec_norm=initvec_norm,
        alphas=alphas,
        betas=betas,
        xs=xs,
        dalphas=dalphas_out,
        dbetas=dbetas_out,
        dxs=dxs_out,
        reortho=True,
        reortho_mode="out",
    )

    dalphas_in = jnp.zeros_like(dalphas)
    dbetas_in = jnp.zeros_like(dbetas)
    output_in = _adjoint_pass(
        matvec=matvec,
        params=params,
        initvec_norm=initvec_norm,
        alphas=alphas,
        betas=betas,
        xs=xs,
        dalphas=dalphas_in,
        dbetas=dbetas_in,
        dxs=dxs_in,
        reortho=True,
        reortho_mode="in",
    )
    return jax.tree_util.tree_map(lambda a, b: a + b, output_out, output_in)


def _adjoint_pass(
    *,
    matvec,
    params,
    initvec_norm,
    alphas,
    betas,
    xs,
    dalphas,
    dbetas,
    dxs,
    reortho,
    reortho_mode=None,
):
    # If reortho_mode is selected, reortho must be true
    if reortho_mode is not None and not reortho:
        raise ValueError

    def adjoint_step(xi_and_lambda, inputs):
        return _adjoint_step(
            *xi_and_lambda,
            matvec=matvec,
            params=params,
            **inputs,
            reortho=reortho,
            reortho_mode=reortho_mode,
        )

    # Scan over all input gradients and output values
    indices = jnp.arange(0, len(xs), step=1)
    xs0 = xs
    xs0 = xs0.at[-1, :].set(jnp.zeros_like(xs[-1, :]))

    loop_over = {
        "dx": dxs[:-1],
        "da": dalphas,
        "db": dbetas,
        "idx": indices[:-1],
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


def _adjoint_step(
    xs_all,
    xi,
    lambda_plus,
    /,
    *,
    matvec,
    params,
    idx,
    dx,
    da,
    db,
    xs,
    a,
    b,
    reortho: bool,
    reortho_mode: str,
):
    # Read inputs
    (xplus, x) = xs

    # Apply formula
    xi /= b
    mu = db - lambda_plus.T @ x + xplus.T @ xi
    nu = da + x.T @ xi
    lambda_ = -xi + mu * xplus + nu * x

    # lambda_ = xs_all.T @ xs_all @ lambda_
    if reortho:
        if reortho_mode == "out":
            zeros = jnp.zeros_like(lambda_)
            xs_all = xs_all.at[idx, :].set(zeros)
            lambda_ = lambda_ - xs_all.T @ (xs_all @ lambda_)
        elif reortho_mode == "in":
            lambda_ = xs_all.T @ xs_all @ lambda_
            # print(lambda_)
            # print(lambda_new)
            # print()
        else:
            msg = "reortho_mode not provided"
            raise ValueError(msg)

    # Value-and-grad of matrix-vector product
    matvec_lambda, vjp = jax.vjp(lambda *p: matvec(lambda_, *p), *params)
    (gradient_increment,) = vjp(x)

    # Prepare next step
    xi = -dx - matvec_lambda + a * lambda_ + b * lambda_plus - b * nu * xplus

    # Return values
    return (xs_all, xi, lambda_), (gradient_increment, lambda_, mu, nu, xi)


def matrix_adjoint(
    *,
    matvec,
    params,
    alphas,
    betas,
    xs,
    residual,
    initlength,
    dalphas,
    dbetas,
    dxs,
    dresidual,
    dinitlength,
):
    # Transpose the inputs (so code matches maths)
    Q = xs.T
    dQ = dxs.T

    # Allocate the multipliers
    Lambda = jnp.zeros_like(Q)
    Sigma = jnp.zeros_like(dQ.T @ Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)

    (A,) = params
    e_1, e_K = jnp.eye(len(alphas))[[0, -1], :]

    # Assemble the dense matrices
    T = _dense_matrix(alphas, betas)
    dT = _dense_matrix(dalphas, dbetas)

    # Solve for gamma
    eta = dT @ e_K - Q.T @ dresidual

    # Solve for L e_K
    Lambda = Lambda.at[:, -1].set(dresidual + Q @ eta)

    # Solve or (Gamma + Gamma.T) e_K
    tmp = jnp.tril(T @ dT.T - (dQ.T @ Q) - Lambda.T @ A @ Q)
    tmp = tmp - 0.5 * jnp.diag(jnp.diag(tmp))
    Gamma = Gamma.at[-1, :].set(tmp[-1, :])

    # Set up the system
    Xi = dQ.T + (Gamma + Gamma.T) @ Q.T + jnp.outer(eta, residual)

    # Solve
    xi = Xi[-1]
    lambda_k = Lambda[:, -1]
    tmp = (xi - (alphas[-1] * lambda_k - A.T @ lambda_k)) / betas[-1]
    Lambda = Lambda.at[:, -2].set(tmp)

    # Return the solution
    return None, (Lambda, jnp.zeros_like(residual), Gamma, Sigma, eta)


def _dense_matrix(diag, off_diag):
    return jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)


def matrix_adjoint_old(
    *, matvec, params, initvec_norm, alphas, betas, xs, dalphas, dbetas, dxs
):
    # Transpose the inputs (so code matches maths)
    Q = xs.T
    dQ = dxs.T

    # Allocate the multipliers
    L = jnp.zeros_like(Q[:, 1:])
    M = jnp.zeros_like(dQ.T @ Q)

    (A,) = params
    e1, e_K = jnp.eye(len(alphas) + 1)[[0, -1], :]

    # Assemble the dense matrices
    T = _dense_matrix_old(alphas, betas)
    dT = _dense_matrix_old(dalphas, dbetas)
    E_K = jnp.eye(len(alphas) + 1)[:, : len(alphas)]

    # Compute M
    rhs = T @ dT.T - dQ.T @ Q
    Xi = jnp.zeros_like(Q.T)
    m = rhs[-1]
    M = M.at[-1].set(m)
    xi = m @ Q.T + dQ[:, -1]
    Xi = Xi.at[-1].set(xi)

    # Initialise the linear-system solve
    lambda_kplus = jnp.zeros_like(Q[:, 0])
    lambda_k = xi / betas[-1]

    # Solve the linear system
    betas_ = jnp.concatenate([jnp.ones((1,)), betas])
    for idx, bminus, a, bplus in zip(
        (jnp.arange(0, len(betas), step=1))[::-1],
        reversed(betas_[:-1]),
        reversed(alphas),
        reversed(betas_[1:]),
    ):
        L = L.at[:, idx].set(lambda_k)

        # Matrix-vector product
        matvec_p = functools.partial(matvec, lambda_k)
        Al, _vjp = jax.vjp(matvec_p, *params)

        # Prepare system
        m = rhs[idx] - Al.T @ Q

        if idx == 0:
            m = m.at[1:].set(0.0)

        M = M.at[idx].set(m)

        xi = m @ Q.T + dQ[:, idx]
        Xi = Xi.at[idx].set(xi)

        # Solve system
        res = xi - bplus * lambda_kplus - a * lambda_k + Al
        lambda_kminus = res / bminus
        lambda_k, lambda_kplus = lambda_kminus, lambda_k

    # Assign the final adjoint variable
    rho = lambda_k

    # Verify a bunch of systems

    def assert_is_small(arr):
        machine_epsilon = 10 * jnp.sqrt(jnp.finfo(jnp.dtype(A)).eps)
        assert jnp.linalg.norm(arr) / jnp.sqrt(arr.size) < machine_epsilon, arr

    # Verify the original system
    residual_original = A @ Q @ E_K - Q @ T
    assert_is_small(residual_original)

    # Verify z_c = 0
    v = Q[:, 0] * initvec_norm
    residual_c = rho.T @ v
    assert_is_small(residual_c)

    # Verify Z_T = 0
    residual_T = L.T @ Q - dT.T
    assert_is_small(residual_T)

    # Verify the solved system
    residual_new = A.T @ L @ E_K.T - L @ T.T - jnp.outer(rho, e1) + Xi.T
    assert_is_small(residual_new)

    # Verify z_Q = 0
    residual_Z = dQ.T + E_K @ L.T @ A - T @ L.T + M @ Q.T - jnp.outer(e1, rho)
    assert_is_small(residual_Z)

    # Verify (Z_Q).T @ Q = 0 (after simplification)
    res1 = dQ.T @ Q + E_K @ L.T @ A @ Q
    res2 = -T @ dT.T + M - jnp.outer(e1, rho) @ Q
    residual_Q = res1 + res2
    assert_is_small(residual_Q)

    # Compute the gradients
    dv = rho * initvec_norm
    dp = L @ E_K.T @ Q.T

    return (dv, dp), (rho, L.T, M, Xi)


def _dense_matrix_old(diag, off_diag):
    k = len(diag)

    # Zero matrix
    T = jnp.zeros((k + 1, k))

    # Diagonal
    T = T.at[:k, :k].set(T[:k, :k] + jnp.diag(diag))

    # Upper diagonal
    T = T.at[:k, :k].set(T[:k, :k] + jnp.diag(off_diag[:-1], 1))

    # Lower diagonal
    return T.at[1:, :k].set(T[1:, :k] + jnp.diag(off_diag))
