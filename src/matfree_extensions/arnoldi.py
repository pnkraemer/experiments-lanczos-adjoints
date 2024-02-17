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


def backward(A, vector, krylov_depth, *, Q, H, r, c, dQ, dH, dr, dc):
    # Transpose the inputs (so code matches maths)

    # Allocate the multipliers
    Lambda = jnp.zeros_like(Q)
    Gamma = jnp.zeros_like(dQ.T @ Q)

    e_1, e_K = jnp.eye(krylov_depth)[[0, -1], :]

    alphas = jnp.diag(H)
    betas = jnp.diag(H, 1)

    # Solve for eta
    eta = dH @ e_K - Q.T @ dr

    # Solve for L e_K
    lambda_k = dr + Q @ eta

    # Initialise
    idx = 1

    # Save result
    Lambda = Lambda.at[:, -idx].set(lambda_k)

    # Solve or (Gamma + Gamma.T) e_K
    e1p = -dc * c * jnp.outer(e_1, e_1)
    tmp = jnp.tril(e1p + H @ dH.T - Lambda.T @ A @ Q - (dQ.T @ Q))
    tmp = tmp - 0.5 * jnp.diag(jnp.diag(tmp))
    Gamma = Gamma.at[-idx, :].set(tmp[-idx, :])

    # Solve for the next lambda
    Xi = dQ.T + (Gamma + Gamma.T) @ Q.T + jnp.outer(eta, r)
    xi = Xi[-idx]

    lambda_kminus = (xi - (alphas[-idx] * lambda_k - A.T @ lambda_k)) / betas[-idx]
    lambda_kplus, lambda_k = lambda_k, lambda_kminus

    betas = jnp.concatenate([jnp.ones((1,)), betas])
    for _ in range(len(alphas) - 1):
        idx += 1
        # Save result
        Lambda = Lambda.at[:, -idx].set(lambda_k)

        # Solve or (Gamma + Gamma.T) e_K
        tmp = jnp.tril(e1p + H @ dH.T - Lambda.T @ A @ Q - (dQ.T @ Q))
        tmp = tmp - 0.5 * jnp.diag(jnp.diag(tmp))
        Gamma = Gamma.at[-idx, :].set(tmp[-idx, :])

        # Solve for the next lambda
        Xi = dQ.T + (Gamma + Gamma.T) @ Q.T + jnp.outer(eta, r)
        xi = Xi[-idx]
        lambda_kminus = (
            xi
            - (alphas[-idx] * lambda_k - A.T @ lambda_k)
            - betas[-(idx - 1)] * lambda_kplus
        ) / betas[-idx]
        lambda_kplus, lambda_k = lambda_k, lambda_kminus

    Sigma = (Lambda.T @ Q - dH.T).T

    # Return the solution
    dv = lambda_k * c
    dA = Lambda @ Q.T
    return (dv, dA), (Lambda, lambda_k, Gamma, Sigma, eta)
