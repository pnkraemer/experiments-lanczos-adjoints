import jax
import jax.numpy as jnp


def arnoldi(matvec, krylov_depth, /, *, reortho: bool, custom_vjp: bool):
    def estimate(v, *params):
        return forward(matvec, v, krylov_depth, *params, reortho=reortho)

    def estimate_fwd(v, *params):
        outputs = estimate(v, *params)
        return outputs, (outputs, params)

    def estimate_bwd(cache, vjp_incoming):
        (Q, H, r, c), params = cache
        dQ, dH, dr, dc = vjp_incoming

        grads, _ = adjoint(
            matvec,
            *params,
            Q=Q,
            H=H,
            r=r,
            c=c,
            dQ=dQ,
            dH=dH,
            dr=dr,
            dc=dc,
            reortho=reortho,
        )
        return grads

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore
    return estimate


def forward(matvec, v, krylov_depth, *params, reortho: bool):
    if krylov_depth < 1 or krylov_depth > len(v):
        msg = f"Parameter depth {krylov_depth} is outside the expected range"
        raise ValueError(msg)

    # Initialise the variables
    (n,), k = jnp.shape(v), krylov_depth
    Q = jnp.zeros((n, k))
    H = jnp.zeros((k, k))
    initlength = jnp.linalg.norm(v)
    init = (Q, H, v, initlength)

    # Fix the step function
    def forward_step(i, val):
        return _forward_step(*val, matvec, *params, idx=i, reortho=reortho)

    # Loop and return
    Q, H, v, _length = jax.lax.fori_loop(0, k, forward_step, init)
    return Q, H, v, 1 / initlength


def _forward_step(Q, H, v, length, matvec, *params, idx, reortho: bool):
    # Save
    v /= length
    Q = Q.at[:, idx].set(v)

    # Evaluate
    v = matvec(v, *params)

    # Orthonormalise
    h = Q.T @ v
    v = v - Q @ h

    # Re-orthonormalise
    if reortho:
        v = v - Q @ (Q.T @ v)

    # Read the length
    length = jnp.linalg.norm(v)

    # Save
    h = h.at[idx + 1].set(length)
    H = H.at[:, idx].set(h)

    return Q, H, v, length


def adjoint(matvec, *params, Q, H, r, c, dQ, dH, dr, dc, reortho: bool):
    # todo: figure out simplifications for symmetric problems

    # Extract the matrix shapes from Q
    nrows, krylov_depth = jnp.shape(Q)

    # Transpose matvec
    def vecmat(x, *p):
        # Use that input shape/dtype == output shape/dtype
        x_like = x

        # Transpose the matrix vector product
        # (as a function of v, not of p)
        vecmat = jax.linear_transpose(lambda s: matvec(s, *p), x_like)

        # The output of the transpose is a tuple of length 1
        (a,) = vecmat(x)
        return a

    # Prepare a bunch of auxiliary matrices

    def lower(m):
        m_tril = jnp.tril(m)
        return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))

    e_1, e_K = jnp.eye(krylov_depth)[[0, -1], :]
    lower_mask = lower(jnp.ones((krylov_depth, krylov_depth)))

    # Initialise
    eta = dH @ e_K - Q.T @ dr
    lambda_k = dr + Q @ eta
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)
    dp = jax.tree_util.tree_map(jnp.zeros_like, *params)

    # Prepare more  auxiliary matrices
    Pi_gamma = -dc * c * jnp.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)
    Pi_xi = dQ.T + jnp.outer(eta, r)

    # The first reorthogonalisation:
    P = Q.T
    ps = dH.T
    ps_mask = jnp.tril(jnp.ones((krylov_depth, krylov_depth)), 1)

    # Loop over those values
    indices = jnp.arange(0, len(H), step=1)
    beta_minuses = jnp.concatenate([jnp.ones((1,)), jnp.diag(H, -1)])
    alphas = jnp.diag(H)
    beta_pluses = H - jnp.diag(jnp.diag(H)) - jnp.diag(jnp.diag(H, -1), -1)
    scan_over = {
        "beta_minus": beta_minuses,
        "alpha": alphas,
        "beta_plus": beta_pluses,
        "idx": indices,
        "lower_mask": lower_mask,
        "Pi_gamma": Pi_gamma,
        "Pi_xi": Pi_xi,
        "p": ps,
        "p_mask": ps_mask,
        "q": Q.T,
    }

    # Fix the step function
    def adjoint_step(x, y):
        output = _adjoint_step(
            *x, **y, vecmat=vecmat, params=params, Q=Q, reortho=reortho
        )
        return output, ()

    # Scan
    init = (lambda_k, Lambda, Gamma, P, dp)
    result, _ = jax.lax.scan(adjoint_step, init, xs=scan_over, reverse=True)
    (lambda_k, Lambda, Gamma, _P, dp) = result

    # Solve for Sigma
    Sigma = (Lambda.T @ Q - dH.T).T

    # Solve for the input gradient
    dv = lambda_k * c

    # Bundle the Lagrange multipliers and return
    multipliers = {
        "Lambda": Lambda,
        "rho": lambda_k,
        "Gamma": Gamma,
        "Sigma": Sigma,
        "eta": eta,
    }
    return (dv, dp), multipliers


def _adjoint_step(
    # Running variables
    lambda_k,
    Lambda,
    Gamma,
    P,
    dp,
    *,
    vecmat,
    params,
    # Loop over: index
    idx,
    # Loop over: submatrices of H
    beta_minus,
    alpha,
    beta_plus,
    # Loop over: auxiliary variables
    lower_mask,
    Pi_gamma,
    Pi_xi,
    q,
    # Loop over: reorthogonalisation
    p,
    p_mask,
    # Fixed parameters
    Q,
    reortho: bool,
):
    # Reorthogonalise
    if reortho:
        # todo: I do not entirely trust the test for this indexing...
        # todo: can we solve for sigma and orthogonalise better?
        p = p * p_mask
        P = P * p_mask[:, None]
        lambda_k = lambda_k - P.T @ (P @ lambda_k) + P.T @ p

    # Save result
    Lambda = Lambda.at[:, idx].set(lambda_k)

    # A single vector-matrix product
    l_At, vjp = jax.vjp(lambda *z: vecmat(lambda_k, *z), *params)
    (dp_increment,) = vjp(q)
    dp = jax.tree_util.tree_map(lambda g, h: g + h, dp, dp_increment)

    # Solve or (Gamma + Gamma.T) e_K
    tmp = lower_mask * (Pi_gamma - l_At @ Q)
    Gamma = Gamma.at[idx, :].set(tmp)

    # Solve for the next lambda
    xi = Pi_xi + (Gamma + Gamma.T)[idx, :] @ Q.T
    asd = beta_plus @ Lambda.T
    lambda_k = (xi - (alpha * lambda_k - l_At) - asd) / beta_minus
    return lambda_k, Lambda, Gamma, P, dp
