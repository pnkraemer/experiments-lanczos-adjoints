"""Implement the adjoint of the Lanczos process."""
import jax
import jax.numpy as jnp
from matfree import test_util


def identity_via_lanczos(*, custom_vjp: bool):
    alg = lanczos_fwd(custom_vjp=custom_vjp)

    def identity(matrix, vec):
        (vecs, diags, offdiags), (v_last, b_last) = alg(matrix, vec)
        dense_matrix = jnp.diag(diags) + jnp.diag(offdiags, 1) + jnp.diag(offdiags, -1)
        return vecs.T @ dense_matrix @ vecs, vec

    return identity


def lanczos_fwd(*, custom_vjp: bool):
    def estimate(matrix, vec):
        small_value = jnp.sqrt(jnp.finfo(jnp.dtype(vec)).eps)

        ((v1, offdiag), diag), v0 = _fwd_init(matrix, vec), vec
        vs, offdiags, diags = [v0], [], []
        vs.append(v1)
        offdiags.append(offdiag)
        diags.append(diag)

        i = 0
        while (offdiag > small_value) and (i := (i + 1)) < len(eigvals):
            ((v1, offdiag), diag), v0 = _fwd_step(A, v1, offdiag, v0), v1

            Qs = jnp.asarray(vs)

            # Reorthogonalisation
            v1 = v1 - Qs.T @ (Qs @ v1)
            v1 /= jnp.linalg.norm(v1)
            vs.append(v1)
            offdiags.append(offdiag)
            diags.append(diag)

        decomp = (jnp.stack(vs[:-1]), jnp.stack(diags), jnp.stack(offdiags[:-1]))
        remainder = (vs[-1], offdiags[-1])
        return decomp, remainder

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

    def estimate_fwd(*args):
        fx = estimate(*args)
        return fx, (fx, A)

    def estimate_bwd(cache, vjp_incoming):
        dxs, dalphas, dbetas = vjp_incoming
        (xs, alphas, betas), A = cache

        k = len(alphas) - 1
        nu_k, lambda_k = _bwd_init(
            dx_Kplus=dxs[k + 1],
            da_K=dalphas[k],
            db_K=dbetas[k],
            b_K=betas[k],
            x_Kplus=xs[k + 1],
            x_K=xs[k],
        )
        lambda_kplus = jnp.zeros_like(lambda_k)
        lambda_kplusplus, lambda_kplus = lambda_kplus, lambda_k

        # a-constraint
        assert jnp.allclose(lambda_kplus.T @ xs[k], dalphas[k])

        # b-constraint
        assert jnp.allclose(lambda_kplus.T @ xs[k + 1], dbetas[k])

        dA = jnp.outer(lambda_kplus, xs[k])

        for k in range(len(alphas) - 1, 0, -1):
            nu_k, lambda_k = _bwd_step(
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
            # a-constraint
            assert jnp.allclose(lambda_kplus.T @ xs[k], dalphas[k])

            # b-constraint
            assert jnp.allclose(
                lambda_kplusplus.T @ xs[k] + lambda_kplus.T @ xs[k + 1], dbetas[k]
            )

            # Update
            dA += jnp.outer(lambda_kplus, xs[k])

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
        return nu_K, lambda_Kplus

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
        return nu_Kminus, lambda_K

    if custom_vjp:
        estimate = jax.custom_vjp(estimate)
        estimate.defvjp(estimate_fwd, estimate_bwd)  # type: ignore

    return estimate


jnp.set_printoptions(3)

# todo: make an identity function
#  to measure whether the autodiff gradients
#  of lanczos make sense


# Set up a test-matrix
eigvals = jnp.ones((10,), dtype=float) * 0.001
eigvals_relevant = jnp.arange(1.0, 2.0, step=0.1)
eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

# Set up an initial vector
key = jax.random.PRNGKey(1)
v = jax.random.normal(key, shape=jnp.shape(eigvals))
v /= jnp.linalg.norm(v)

algorithm = identity_via_lanczos(custom_vjp=False)
(A_like, v_like), vjp = jax.vjp(algorithm, A, v)
# print(A_like, v_like)
# A_like, v_like = algorithm(A, v)

print(A_like - A)
assert False
assert jnp.allclose(A_like, A)
assert jnp.allclose(v_like, v)
assert False


print("Autodiff VJP:")
M, init = vjp((A_like, v_like))
print(M)
print(A)
# print(init)
print()


print("Custom VJP:")
algorithm = identity_via_lanczos(custom_vjp=True)
(A_like, v_like), vjp = jax.vjp(algorithm, A, v)
assert jnp.allclose(A_like, A)
assert jnp.allclose(v_like, v)
M, init = vjp((A_like, v_like))
print(M)
print(A)
# print(init)
