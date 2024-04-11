import jax
import jax.numpy as jnp


def hessenberg(matvec, krylov_depth, /, *, reortho: str, custom_vjp: bool = True):
    def estimate(v, *params):
        return forward(matvec, krylov_depth, v, *params, reortho=reortho)

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


def forward(matvec, krylov_depth, v, *params, reortho: str):
    if krylov_depth < 1 or krylov_depth > len(v):
        msg = f"Parameter depth {krylov_depth} is outside the expected range"
        raise ValueError(msg)

    reortho_expected = ["none", "full"]
    if not isinstance(reortho, str) or reortho not in reortho_expected:
        msg = f"Unexpected input for {reortho}: either of {reortho_expected} expected."
        raise TypeError(msg)

    # Initialise the variables
    (n,), k = jnp.shape(v), krylov_depth
    Q = jnp.zeros((n, k), dtype=v.dtype)
    H = jnp.zeros((k, k), dtype=v.dtype)
    initlength = jnp.sqrt(jnp.dot(v.conj(), v))
    init = (Q, H, v, initlength)

    # Fix the step function
    def forward_step(i, val):
        return _forward_step(*val, matvec, *params, idx=i, reortho=reortho)

    # Loop and return
    Q, H, v, _length = jax.lax.fori_loop(0, k, forward_step, init)
    return Q, H, v, 1 / initlength


def _forward_step(Q, H, v, length, matvec, *params, idx, reortho: str):
    # Save
    v /= length
    Q = Q.at[:, idx].set(v)

    # Evaluate
    v = matvec(v, *params)

    # Orthonormalise
    h = Q.T.conj() @ v
    v = v - Q @ h

    # Re-orthonormalise
    if reortho != "none":
        v = v - Q @ (Q.T.conj() @ v)

    # Read the length
    length = jnp.sqrt(jnp.dot(v.conj(), v))

    # Save
    h = h.at[idx + 1].set(length)
    H = H.at[:, idx].set(h)

    return Q, H, v, length


def adjoint(matvec, *params, Q, H, r, c, dQ, dH, dr, dc, reortho: str):
    # todo: implement simplifications for symmetric problems

    reortho_expected = ["none", "full", "full_with_sigma"]
    if not isinstance(reortho, str) or reortho not in reortho_expected:
        msg = f"Unexpected input for {reortho}: either of {reortho_expected} expected."
        raise TypeError(msg)

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
        return a.conj()

    # Prepare a bunch of auxiliary matrices

    def lower(m):
        m_tril = jnp.tril(m)
        return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))

    e_1, e_K = jnp.eye(krylov_depth)[[0, -1], :]
    lower_mask = lower(jnp.ones((krylov_depth, krylov_depth)))

    # Initialise
    eta = dH @ e_K - Q.T.conj() @ dr
    lambda_k = dr + Q @ eta
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)
    Sigma = jnp.zeros_like(dQ.T @ Q)
    dp = jax.tree_util.tree_map(jnp.zeros_like, *params) if params != () else ()

    # Prepare more  auxiliary matrices
    Pi_xi = dQ.T + jnp.outer(eta, r)
    Pi_gamma = (
        -dc * c.conj() * jnp.outer(e_1, e_1) + H.conj() @ dH.T - (dQ.T.conj() @ Q)
    )

    # Prepare reorthogonalisation:
    P = Q.T
    ps = dH.T
    ps_mask = jnp.tril(jnp.ones((krylov_depth, krylov_depth)), 1)

    # Prepare fancy reorthogonalisation
    Pi_sigma = dQ.T @ Q - H @ dH.T
    Pi_sigma_mask = jnp.triu(jnp.ones((krylov_depth, krylov_depth)), 1)
    H_padded = jnp.eye(len(Sigma), dtype=H.dtype)
    H_padded = H_padded.at[1:-1, 1:-1].set(H[1:-1, :-2])

    # Loop over those values
    indices = jnp.arange(0, len(H), step=1)
    beta_minuses = jnp.concatenate([jnp.ones((1,)), jnp.diag(H, -1)])
    alphas = jnp.diag(H)
    beta_pluses = H - jnp.diag(jnp.diag(H)) - jnp.diag(jnp.diag(H, -1), -1)
    # todo: the number of loop-variables is getting out of hand...
    scan_over = {
        "beta_minus": beta_minuses,
        "alpha": alphas,
        "beta_plus": beta_pluses,
        "idx": indices,
        "lower_mask": lower_mask,
        "Pi_gamma": Pi_gamma,
        "Pi_xi": Pi_xi,
        "Pi_sigma": Pi_sigma,
        "Pi_sigma_mask": Pi_sigma_mask,
        "h_padded": H_padded,
        "p": ps,
        "p_mask": ps_mask,
        "q": Q.T.conj(),
    }

    # Fix the step function
    def adjoint_step(x, y):
        output = _adjoint_step(
            *x, **y, vecmat=vecmat, params=params, Q=Q, reortho=reortho
        )
        return output, ()

    # Scan
    sigma_init = jnp.zeros((krylov_depth,), dtype=lambda_k.dtype)
    init = (lambda_k, Lambda, Gamma, Sigma, P, dp, sigma_init)
    result, _ = jax.lax.scan(adjoint_step, init, xs=scan_over, reverse=True)
    (lambda_k, Lambda, Gamma, Sigma_t, _P, dp, _sigma) = result

    # Finalise Sigma
    if reortho == "full_with_sigma":
        Sigma_t = jnp.roll(Sigma_t, -1, axis=0)
    else:
        Sigma_t = Lambda.T @ Q - dH.T

    # Solve for the input gradient
    dv = lambda_k * c

    # Bundle the Lagrange multipliers and return
    multipliers = {
        "Lambda": Lambda,
        "rho": lambda_k,
        "Gamma": Gamma,
        "Sigma": jnp.triu(Sigma_t, 2).T,
        "eta": eta,
    }
    return (dv, dp), multipliers


def _adjoint_step(
    # Running variables
    lambda_k,
    Lambda,
    Gamma,
    Sigma,
    P,
    dp,
    sigma,
    *,
    # Matrix-vector product
    vecmat,
    params,
    # Loop over: index
    idx,
    # Loop over: submatrices of H
    beta_minus,
    alpha,
    beta_plus,
    # Loop over: auxiliary variables for Gamma and Sigma
    lower_mask,
    Pi_gamma,
    Pi_xi,
    Pi_sigma,
    Pi_sigma_mask,
    h_padded,
    q,
    # Loop over: reorthogonalisation
    p,
    p_mask,
    # Other parameters
    Q,
    reortho: str,
):
    # Reorthogonalise
    if reortho != "none":
        if reortho == "full_with_sigma":
            p = p + sigma
        elif reortho == "full":
            P = p_mask[:, None] * P
            p = p_mask * p
        else:
            raise ValueError
        lambda_k = lambda_k - P.T @ (P @ lambda_k) + P.T @ p

    # A single vector-matrix product
    l_At, vjp = jax.vjp(lambda *z: vecmat(lambda_k, *z), *params)

    # Update the parameter-gradients
    dp = jax.tree_util.tree_map(lambda g, h: g + h, dp, *vjp(q)) if params != () else ()

    # Solve or (Gamma + Gamma.T) e_K
    tmp = lower_mask * (Pi_gamma - l_At @ Q.conj())
    Gamma = Gamma.at[idx, :].set(tmp)

    # Solve for the next Sigma
    sigma_ = Pi_sigma_mask * (Pi_sigma + l_At @ Q + (Gamma + Gamma.T)[idx, :])
    sigma = sigma_ - h_padded @ Sigma
    sigma /= beta_minus

    # Save Lambda and Sigma
    Lambda = Lambda.at[:, idx].set(lambda_k)
    Sigma = Sigma.at[idx, :].set(sigma)

    # Solve for the next lambda
    xi = Pi_xi + (Gamma + Gamma.T)[idx, :] @ Q.T
    asd = beta_plus @ Lambda.T
    lambda_k = xi - (alpha * lambda_k - l_At) - asd
    lambda_k /= beta_minus
    return lambda_k, Lambda, Gamma, Sigma, P, dp, sigma
