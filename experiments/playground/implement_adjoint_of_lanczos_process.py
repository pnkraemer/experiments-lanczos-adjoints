"""Implement the adjoint of the Lanczos process."""
import jax
import jax.numpy as jnp
from matfree import test_util


def lanczos_fwd(*, custom_vjp: bool):
    def _fwd_init(matrix, vec):
        """Initialize Lanczos' algorithm.

        Solve A x_{k} = a_k x_k + b_k x_{k+1}
        for x_{k+1}, a_k, and b_k, using
        orthogonality of the x_k.
        """
        a = vec @ (matrix @ vec)
        r = matrix @ vec - a * vec
        b = jnp.linalg.norm(r)
        x = r / b
        return (x, b), a

    def _fwd_step(matrix, vec, b, vec_previous):
        """Apply Lanczos' recurrence.

        Solve
        A x_{k} = b_{k-1} x_{k-1} + a_k x_k + b_k x_{k+1}
        for x_{k+1}, a_k, and b_k, using
        orthogonality of the x_k.
        """
        a = vec @ (matrix @ vec)
        r = matrix @ vec - a * vec - b * vec_previous
        b = jnp.linalg.norm(r)
        x = r / b
        return (x, b), a

    def estimate(matrix, vec):
        small_value = jnp.sqrt(jnp.finfo(jnp.dtype(vec)).eps)

        ((v1, offdiag), diag), v0 = _fwd_init(matrix, vec), vec
        vs, offdiags, diags = [v0], [], []

        i = 0
        while (offdiag > small_value) and (i := (i + 1)) < len(eigvals):
            vs.append(v1)
            offdiags.append(offdiag)
            diags.append(diag)
            ((v1, offdiag), diag), v0 = _fwd_step(A, v1, offdiag, v0), v1

        return jnp.stack(vs), jnp.stack(diags), jnp.stack(offdiags)

    def estimate_fwd(*args):
        fx = estimate(*args)
        return fx, (fx, A)

    def estimate_bwd(cache, vjp_incoming):
        dxs, dalphas, dbetas = vjp_incoming
        (xs, alphas, betas), A = cache

        k = len(alphas) - 1
        mu_k, nu_k, lambda_kplus = _bwd_init(
            dx_Kplus=dxs[k + 1],
            da_K=dalphas[k],
            db_K=dbetas[k],
            b_K=betas[k],
            x_Kplus=xs[k + 1],
            x_K=xs[k],
        )
        lambda_kplusplus = jnp.zeros_like(lambda_kplus)
        dA = jnp.outer(lambda_kplus, xs[k])
        for k in range(len(alphas) - 1, 0, -1):
            mu_k, nu_k, lambda_k = _bwd_step(
                A=A,
                dx_K=dxs[k],
                da_Kminus=dalphas[k - 1],
                db_Kminus=dbetas[k - 1],
                lambda_kplus=lambda_kplus,
                a_K=alphas[k],
                b_K=betas[k],
                lambda_Kplusplus=lambda_kplusplus,
                b_Kminus=betas[k - 1],
                nu_K=nu_k,
                x_Kplus=xs[k + 1],
                x_K=xs[k],
                x_Kminus=xs[k - 1],
            )
            dA += jnp.outer(lambda_k, xs[k])
            lambda_kplusplus, lambda_kplus = lambda_kplus, lambda_k
        lambda_1 = (
            betas[0] * lambda_kplusplus
            - A.T @ lambda_kplus
            + alphas[0] * lambda_kplus
            - nu_k * xs[1]
        )
        dv = -lambda_1
        return dA, dv

    def _bwd_init(*, dx_Kplus, da_K, db_K, b_K, x_Kplus, x_K):
        mu_K = db_K * b_K - x_Kplus.T @ dx_Kplus
        nu_K = da_K * b_K - x_K.T @ dx_Kplus
        lambda_Kplus = (dx_Kplus + mu_K * x_Kplus + nu_K * x_K) / b_K
        return mu_K, nu_K, lambda_Kplus

    def _bwd_step(
        *,
        A,
        dx_K,
        db_Kminus,
        da_Kminus,
        lambda_kplus,
        a_K,
        b_K,
        lambda_Kplusplus,
        b_Kminus,
        nu_K,
        x_Kplus,
        x_K,
        x_Kminus,
    ):
        xi = (
            dx_K
            + A.T @ lambda_kplus
            - a_K * lambda_kplus
            - b_K * lambda_Kplusplus
            + nu_K * x_Kplus
        )
        mu_Kminus = b_Kminus * (db_Kminus - lambda_kplus @ x_Kminus) - x_K.T @ xi
        nu_Kminus = b_Kminus * da_Kminus - x_Kminus.T @ xi
        lambda_K = (xi + mu_Kminus * x_K + nu_Kminus * x_Kminus) / b_Kminus
        return mu_Kminus, nu_Kminus, lambda_K

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


jnp.set_printoptions(3)

# Set up a test-matrix
eigvals = jnp.ones((50,), dtype=float) * 0.001
eigvals_relevant = jnp.arange(1.0, 2.0, step=0.1)
eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

# Set up an initial vector
key = jax.random.PRNGKey(1)
v = jax.random.normal(key, shape=jnp.shape(eigvals))
v /= jnp.linalg.norm(v)

# Run algorithm
algorithm = lanczos_fwd(custom_vjp=True)
(vecs, diags, offdiags), vjp = jax.vjp(algorithm, A, v)

M, init = vjp((vecs, diags, offdiags))
# print(M)
print(init)
print()

algorithm = lanczos_fwd(custom_vjp=False)
(vecs, diags, offdiags), vjp = jax.vjp(algorithm, A, v)

M, init = vjp((vecs, diags, offdiags))
# print(M)
print(init)
