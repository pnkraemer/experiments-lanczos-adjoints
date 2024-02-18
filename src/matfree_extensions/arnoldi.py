import jax
import jax.numpy as jnp


def forward(A, v, krylov_depth):
    if krylov_depth < 1 or krylov_depth > len(v):
        msg = f"Parameter depth {krylov_depth} is outside the expected range"
        raise ValueError(msg)

    # Initialise the variables
    (n,), k = jnp.shape(v), krylov_depth
    Q = jnp.zeros((n, k))
    H = jnp.zeros((k, k))
    initlength = jnp.linalg.norm(v)
    init = (Q, H, v, initlength)

    if krylov_depth == 0:
        return Q, H, v, 1 / initlength

    # Fix the step function
    def forward_step(i, val):
        return _forward_step(*val, A=A, idx=i)

    # Loop and return
    Q, H, v, _length = jax.lax.fori_loop(0, k, forward_step, init)
    return Q, H, v, 1 / initlength


def _forward_step(Q, H, v, length, *, A, idx):
    # Save
    v /= length
    Q = Q.at[:, idx].set(v)

    # Evaluate
    v = A @ v

    # Orthonormalise
    h = Q.T @ v
    v = v - Q @ h
    length = jnp.linalg.norm(v)

    # Save
    h = h.at[idx + 1].set(length)
    H = H.at[:, idx].set(h)

    return Q, H, v, length


def vjp(A, krylov_depth, *, Q, H, r, c, dQ, dH, dr, dc):
    tmp = adjoint(A, krylov_depth, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc)
    Lambda, lambda_k, _Gamma, _Sigma, _eta = tmp

    # Return the solution
    dv = lambda_k * c
    dA = Lambda @ Q.T
    return dA, dv


def adjoint(A, krylov_depth, *, Q, H, r, c, dQ, dH, dr, dc):
    # Prepare a bunch of auxiliary matrices
    e_1, e_K = jnp.eye(krylov_depth)[[0, -1], :]
    Pi = -dc * c * jnp.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)

    # Initialise
    eta = dH @ e_K - Q.T @ dr
    lambda_k = dr + Q @ eta
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)

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
    }

    # Fix the step function
    def adjoint_step(x, y):
        carry = _adjoint_step(*x, **y, Pi=Pi, A=A, Q=Q, dQ=dQ, eta=eta, r=r)
        return carry, ()

    # Scan
    init = (lambda_k, Lambda, Gamma)
    result, _ = jax.lax.scan(adjoint_step, init, xs=scan_over, reverse=True)
    (lambda_k, Lambda, Gamma) = result

    # Solve for Sigma
    Sigma = (Lambda.T @ Q - dH.T).T

    # Return the results
    return Lambda, lambda_k, Gamma, Sigma, eta


def _adjoint_step(
    lambda_k, Lambda, Gamma, *, idx, beta_minus, alpha, beta_plus, Pi, A, Q, dQ, eta, r
):
    # todo: replace all Lambda.T @ A calls with lambda_k @ A
    #
    # todo: do we need to assemble Gamma via matrix-matrix arithmetic?
    #  or can we use vector arithmetic and yield the row of gamma?
    #
    # todo: can we assemble a row of Xi with vector-arithmetic
    #  instead of matrix arithmetic?
    #
    # todo: do we need the full Pi?

    # Save result
    Lambda = Lambda.at[:, idx].set(lambda_k)

    # Solve or (Gamma + Gamma.T) e_K
    tmp = _lower(Pi - Lambda.T @ A @ Q)
    Gamma = Gamma.at[idx, :].set(tmp[idx, :])

    # Solve for the next lambda
    Xi = dQ.T + (Gamma + Gamma.T) @ Q.T + jnp.outer(eta, r)
    xi = Xi[idx]
    asd = beta_plus @ Lambda.T
    lambda_k = (xi - (alpha * lambda_k - A.T @ lambda_k) - asd) / beta_minus
    return lambda_k, Lambda, Gamma


def _lower(m):
    m_tril = jnp.tril(m)
    return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))
