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
        vs, offdiags, diags = [v1], [offdiag], [diag]

        i = 0
        while (offdiag > small_value) and (i := (i + 1)) < len(eigvals):
            ((v1, offdiag), diag), v0 = _fwd_step(A, v1, offdiag, v0), v1
            vs.append(v1)
            offdiags.append(offdiag)
            diags.append(diag)

        return jnp.stack(vs), jnp.stack(diags), jnp.stack(offdiags)

    def estimate_fwd(*args):
        fx = estimate(*args)
        return fx, (fx, A)

    def estimate_bwd(cache, vjp_incoming):
        dxs, dalphas, dbetas = vjp_incoming
        (xs, alphas, betas), A = cache

        mu_k, nu_k, lambda_kplus = _bwd_init(
            dx_K=dxs[-1],
            da_K=dalphas[-1],
            db_K=dbetas[-1],
            b_K=betas[-1],
            x_Kplus=xs[-1],
            x_K=xs[-2],
        )
        dA = jnp.outer(lambda_kplus, xs[-1])

        mu_kminus, nu_kminus, lambda_k = _bwd_step(
            A=A,
            dx_Kminus=dxs[-2],
            da_Kminus=dalphas[-2],
            db_Kminus=dbetas[-2],
            lambda_kplus=lambda_kplus,
            a_K=alphas[-1],
            b_K=betas[-1],
            lambda_Kplusplus=jnp.zeros_like(lambda_kplus),
            b_Kminus=betas[-2],
            nu_K=nu_k,
            x_Kplus=xs[-1],
            x_K=xs[-2],
            x_Kminus=xs[-3],
        )
        dA += jnp.outer(lambda_k, xs[-2])

        dv = lambda_kplus
        return dA, dv

    def _bwd_init(*, dx_K, da_K, db_K, b_K, x_Kplus, x_K):
        mu_K = 0.5 * (db_K * b_K - x_Kplus.T @ dx_K)
        nu_K = da_K * b_K - x_K.T @ dx_K
        lambda_Kplus = (dx_K + 2 * mu_K * x_Kplus + nu_K * x_K) / b_K
        return mu_K, nu_K, lambda_Kplus

    def _bwd_step(
        *,
        A,
        dx_Kminus,
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
            dx_Kminus
            + A.T @ lambda_kplus
            - a_K * lambda_kplus
            + nu_K * x_Kplus
            - b_K * lambda_Kplusplus
        )
        mu_Kminus = 0.5 * (
            b_Kminus * (db_Kminus - lambda_kplus @ x_Kminus) - x_K.T @ xi
        )
        nu_Kminus = b_Kminus * da_Kminus - x_Kminus.T @ xi
        lambda_K = (xi + 2 * mu_Kminus * x_K + nu_Kminus * x_Kminus) / b_Kminus
        return mu_Kminus, nu_Kminus, lambda_K

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


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
print(M)
print(init)
