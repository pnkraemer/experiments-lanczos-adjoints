from typing import Callable

import jax
import jax.numpy as jnp


def hessenberg(matvec, krylov_depth, /, *, reortho: str, custom_vjp: bool = True):
    reortho_expected = ["none", "full"]
    if reortho not in reortho_expected:
        msg = f"Unexpected input for {reortho}: either of {reortho_expected} expected."
        raise TypeError(msg)

    def estimate_public(v, *params):
        matvec_convert, aux_args = jax.closure_convert(matvec, v, *params)
        return estimate_backend(matvec_convert, v, *params, *aux_args)

    def estimate_backend(matvec_convert: Callable, v, *params):
        return _forward(matvec_convert, krylov_depth, v, *params, reortho=reortho)

    def estimate_fwd(matvec_convert: Callable, v, *params):
        outputs = estimate_backend(matvec_convert, v, *params)
        return outputs, (outputs, params)

    def estimate_bwd(matvec_convert: Callable, cache, vjp_incoming):
        (Q, H, r, c), params = cache
        dQ, dH, dr, dc = vjp_incoming

        return _adjoint(
            matvec_convert,
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

    if custom_vjp:
        estimate_backend = jax.custom_vjp(estimate_backend, nondiff_argnums=(0,))
        estimate_backend.defvjp(estimate_fwd, estimate_bwd)  # type: ignore
    return estimate_public


def _forward(matvec, krylov_depth, v, *params, reortho: str):
    if krylov_depth < 1 or krylov_depth > len(v):
        msg = f"Parameter depth {krylov_depth} is outside the expected range"
        raise ValueError(msg)

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


def _adjoint(matvec, *params, Q, H, r, c, dQ, dH, dr, dc, reortho: str):
    # todo: implement simplifications for symmetric problems

    # Extract the matrix shapes from Q
    _, krylov_depth = jnp.shape(Q)

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
    Pi_xi = dQ.T + jnp.outer(eta, r)
    Pi_gamma = -dc * c * jnp.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)

    # Prepare reorthogonalisation:
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
            *x, **y, matvec=matvec, params=params, Q=Q, reortho=reortho
        )
        return output, ()

    # Scan
    init = (lambda_k, Lambda, Gamma, P, dp)
    result, _ = jax.lax.scan(adjoint_step, init, xs=scan_over, reverse=True)
    (lambda_k, Lambda, Gamma, _P, dp) = result

    # Solve for the input gradient
    dv = lambda_k * c

    return dv, dp


def _adjoint_step(
    # Running variables
    lambda_k,
    Lambda,
    Gamma,
    P,
    dp,
    *,
    # Matrix-vector product
    matvec,
    params,
    # Loop over: index
    idx,
    # Loop over: submatrices of H
    beta_minus,
    alpha,
    beta_plus,
    # Loop over: auxiliary variables for Gamma
    lower_mask,
    Pi_gamma,
    Pi_xi,
    q,
    # Loop over: reorthogonalisation
    p,
    p_mask,
    # Other parameters
    Q,
    reortho: str,
):
    # Reorthogonalise
    if reortho == "full":
        P = p_mask[:, None] * P
        p = p_mask * p
        lambda_k = lambda_k - P.T @ (P @ lambda_k) + P.T @ p

    # Transposed matvec and parameter-gradient in a single matvec
    _, vjp = jax.vjp(matvec, q, *params)
    l_At, dp_increment = vjp(lambda_k)
    dp = jax.tree_util.tree_map(lambda g, h: g + h, dp, dp_increment)

    # Solve for (Gamma + Gamma.T) e_K
    tmp = lower_mask * (Pi_gamma - l_At @ Q)
    Gamma = Gamma.at[idx, :].set(tmp)

    # Solve for the next lambda (backward substitution step)
    Lambda = Lambda.at[:, idx].set(lambda_k)
    xi = Pi_xi + (Gamma + Gamma.T)[idx, :] @ Q.T
    lambda_k = xi - (alpha * lambda_k - l_At) - beta_plus @ Lambda.T
    lambda_k /= beta_minus
    return lambda_k, Lambda, Gamma, P, dp
