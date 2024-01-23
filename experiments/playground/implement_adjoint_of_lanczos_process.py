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
        return fx, fx

    def estimate_bwd(cache, vjp_incoming):
        dxs, dalphas, dbetas = vjp_incoming
        (xs, alphas, betas) = cache

        mu_k, nu_k, lambda_kplus = _bwd_init(
            dxs[-1], dalphas[-1], dbetas[-1], betas[-1], xs[-1], xs[-2]
        )

        dA = jnp.outer(lambda_kplus, xs[-1])
        dv = lambda_kplus
        return dA, dv

    def _bwd_init(dx, da, db, b, xplus, x):
        mu = 0.5 * (db * b - xplus.T @ dx)
        nu = da * b - x.T @ dx
        lambda_ = (dx + 2 * mu * xplus + nu * x) / b
        return mu, nu, lambda_

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
