import jax.numpy as jnp


def forward(A, v, krylov_depth):
    (n,) = jnp.shape(v)
    k = krylov_depth

    H = jnp.zeros((k, k))
    Q = jnp.zeros((n, k))

    initlength = jnp.linalg.norm(v)
    v /= initlength

    for i in range(k):
        # Save
        Q = Q.at[:, i].set(v)

        # Evaluate
        v = A @ v

        # Orthonormalise
        h = Q.T @ v
        v = v - Q @ h
        length = jnp.linalg.norm(v)
        v /= length

        # Save
        h = h.at[i + 1].set(length)
        H = H.at[:, i].set(h)

    return Q, H, v * length, 1 / initlength


def vjp(A, krylov_depth, *, Q, H, r, c, dQ, dH, dr, dc):
    tmp = adjoint(A, krylov_depth, Q=Q, H=H, r=r, c=c, dQ=dQ, dH=dH, dr=dr, dc=dc)
    Lambda, lambda_k, _Gamma, _Sigma, _eta = tmp

    # Return the solution
    dv = lambda_k * c
    dA = Lambda @ Q.T
    return dA, dv


def adjoint(A, krylov_depth, *, Q, H, r, c, dQ, dH, dr, dc):
    # Allocate some needed matrices
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)
    e_1, e_K = jnp.eye(krylov_depth)[[0, -1], :]

    # Set up extended linear system
    H_extended = jnp.zeros((len(H) + 1, len(H) + 1))
    H_extended = H_extended.at[:-1, 1:].set(H)
    H_extended = H_extended.at[0, 0].set(1.0)
    H_extended = H_extended.at[-1, -1].set(1.0)

    # Solve for eta
    eta = dH @ e_K - Q.T @ dr

    # Solve for L e_K
    lambda_k = dr + Q @ eta

    # Initialise
    idx = 1

    # Save result
    Lambda = Lambda.at[:, -idx].set(lambda_k)

    # Solve for (Gamma + Gamma.T) e_K
    Pi = -dc * c * jnp.outer(e_1, e_1) + H @ dH.T - (dQ.T @ Q)
    tmp = _lower(Pi - Lambda.T @ A @ Q)
    Gamma = Gamma.at[-idx, :].set(tmp[-idx, :])

    # Solve for the next lambda
    Xi = dQ.T + (Gamma + Gamma.T) @ Q.T + jnp.outer(eta, r)
    xi = Xi[-idx]

    # Initialise the iteration
    beta = H_extended[-(idx + 1), -(idx + 1)]
    alpha = H_extended[-(idx + 1), -idx]
    beta_plus = H_extended[-(idx + 1), 1:]
    beta_plus = beta_plus.at[-idx].set(0.0)
    beta_plus = beta_plus.at[-(idx + 1)].set(0.0)
    asd = beta_plus @ Lambda.T
    lambda_k = (xi - (alpha * lambda_k - A.T @ lambda_k) - asd) / beta

    for _ in range(len(H) - 1):
        idx += 1
        # Save result
        Lambda = Lambda.at[:, -idx].set(lambda_k)

        # Read scalar coefficeints
        beta_minus = H_extended[-(idx + 1), -(idx + 1)]
        alpha = H_extended[-(idx + 1), -idx]

        # Read remainind coefficients
        beta_plus = H_extended[-(idx + 1), 1:]
        beta_plus = beta_plus.at[-idx].set(0.0)
        beta_plus = beta_plus.at[-(idx + 1)].set(0.0)

        # Solve or (Gamma + Gamma.T) e_K
        tmp = _lower(Pi - Lambda.T @ A @ Q)
        Gamma = Gamma.at[-idx, :].set(tmp[-idx, :])

        # Solve for the next lambda
        Xi = dQ.T + (Gamma + Gamma.T) @ Q.T + jnp.outer(eta, r)
        xi = Xi[-idx]
        asd = beta_plus @ Lambda.T
        lambda_k = (xi - (alpha * lambda_k - A.T @ lambda_k) - asd) / beta_minus

    # Solve for Sigma
    Sigma = (Lambda.T @ Q - dH.T).T

    return (Lambda, lambda_k, Gamma, Sigma, eta)


def _lower(m):
    m_tril = jnp.tril(m)
    return m_tril - 0.5 * jnp.diag(jnp.diag(m_tril))
