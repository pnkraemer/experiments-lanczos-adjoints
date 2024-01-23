"""Implement the adjoint of the Lanczos process."""
import jax
import jax.flatten_util
import jax.numpy as jnp
from matfree import test_util


def sym(x):
    return jnp.triu(x) - jnp.diag(0.5 * jnp.diag(x))


def identity_via_lanczos(matvec, *, custom_vjp: bool):
    alg = lanczos_fwd(matvec, custom_vjp=custom_vjp)

    def identity(vec, *params):
        (vecs, diags, offdiags), _ = alg(vec, *params)
        v0 = vecs[0]

        dense_matrix = jnp.diag(diags) + jnp.diag(offdiags, 1) + jnp.diag(offdiags, -1)
        return v0, (vecs.T @ dense_matrix @ vecs)

    return identity


def lanczos_fwd(matvec, *, custom_vjp: bool):
    def estimate(vec, *params):
        small_value = jnp.sqrt(jnp.finfo(jnp.dtype(vec)).eps)

        ((v1, offdiag), diag), v0 = _fwd_init(vec, *params), vec
        vs, offdiags, diags = [v0], [], []

        # Store results
        vs.append(v1)
        offdiags.append(offdiag)
        diags.append(diag)

        i = 0
        while (offdiag > small_value) and (i := (i + 1)) < len(eigvals):
            ((v1, offdiag), diag), v0 = _fwd_step(v1, offdiag, v0, *params), v1

            # Reorthogonalisation
            Qs = jnp.asarray(vs)
            v1 = v1 - Qs.T @ (Qs @ v1)
            v1 /= jnp.linalg.norm(v1)

            # Store results
            vs.append(v1)
            offdiags.append(offdiag)
            diags.append(diag)

        decomp = (jnp.asarray(vs[:-1]), jnp.asarray(diags), jnp.asarray(offdiags[:-1]))
        remainder = (vs[-1], offdiags[-1])
        return decomp, remainder

    def _fwd_init(vec, *params):
        """Initialize Lanczos' algorithm.

        Solve A x_{k} = a_k x_k + b_k x_{k+1}
        for x_{k+1}, a_k, and b_k, using
        orthogonality of the x_k.
        """
        a = vec @ (matvec(vec, *params))
        r = (matvec(vec, *params)) - a * vec
        b = jnp.linalg.norm(r)
        x = r / b
        return (x, b), a

    def _fwd_step(vec, b, vec_previous, *params):
        """Apply Lanczos' recurrence.

        Solve
        A x_{k} = b_{k-1} x_{k-1} + a_k x_k + b_k x_{k+1}
        for x_{k+1}, a_k, and b_k, using
        orthogonality of the x_k.
        """
        a = vec @ (matvec(vec, *params))
        r = matvec(vec, *params) - a * vec - b * vec_previous
        b = jnp.linalg.norm(r)
        x = r / b
        return (x, b), a

    def estimate_fwd(vec, *params):
        fx = estimate(vec, *params)
        return fx, (fx, *params)

    def estimate_bwd(cache, vjp_incoming):
        (dxs, dalphas, dbetas), (dx_last, dbeta_last) = vjp_incoming

        # todo (to debug the dA value):
        #  check that _all_ input gradients are used! (common error)
        #  verify the maths
        #  once the gradients are the same
        #  (or at least they would lead to the correct reduction
        #  after summing up the rows/columns in the jacobian matrix)
        #  derive the parameter-derivative (on paper) so we can use
        #  the correct derivative wrt a _symmetric_ matrix (not any matrix)

        dxs = jnp.concatenate((dxs, dx_last[None]))
        dbetas = jnp.concatenate((dbetas, dbeta_last[None]))

        ((xs, alphas, betas), (x_last, beta_last)), *params = cache
        xs = jnp.concatenate((xs, x_last[None]))
        betas = jnp.concatenate((betas, beta_last[None]))

        k = len(alphas) - 1
        nu_k, lambda_k = _bwd_init(
            dx_Kplus=dxs[k + 1],
            da_K=dalphas[k],
            db_K=dbetas[k],
            b_K=betas[k],
            x_Kplus=xs[k + 1],
            x_K=xs[k],
        )
        dA = jnp.outer(lambda_k, xs[k])
        lambda_kplus = jnp.zeros_like(lambda_k)
        lambda_kplusplus, lambda_kplus = lambda_kplus, lambda_k

        # a-constraint
        assert jnp.allclose(lambda_kplus.T @ xs[k], dalphas[k])

        # b-constraint
        assert jnp.allclose(lambda_kplus.T @ xs[k + 1], dbetas[k])

        for k in range(len(alphas) - 1, 0, -1):
            nu_k, lambda_k = _bwd_step(
                *params,
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
            dA += jnp.outer(lambda_k, xs[k - 1])

            lambda_kplusplus, lambda_kplus = lambda_kplus, lambda_k

        lambda_1 = (
            -dxs[0]
            + betas[0] * lambda_kplusplus
            - matvec(lambda_kplus, *params)
            + alphas[0] * lambda_kplus
            - nu_k * xs[1]
        )
        dv = -lambda_1
        return dv, dA

    def _bwd_init(dx_Kplus, da_K, db_K, b_K, x_Kplus, x_K):
        mu_K = db_K * b_K - x_Kplus.T @ dx_Kplus
        nu_K = da_K * b_K - x_K.T @ dx_Kplus
        lambda_Kplus = (dx_Kplus + mu_K * x_Kplus + nu_K * x_K) / b_K
        return nu_K, lambda_Kplus

    def _bwd_step(
        *params,
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
            + matvec(lambda_kplus, *params)
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


jnp.set_printoptions(4)

# todo: verify the maths, and then revisit to verify the implementation.

# Set up a test-matrix
eigvals = jnp.ones((2,), dtype=float) * 0.001
eigvals_relevant = jnp.arange(1.0, 2.0, step=1.0 / len(eigvals))
eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

# Set up an initial vector
key = jax.random.PRNGKey(1)
v = jax.random.normal(key, shape=jnp.shape(eigvals))
v /= jnp.linalg.norm(v)


algorithm = identity_via_lanczos(lambda s, p: p @ s, custom_vjp=False)
# A = sym(A)

# So is the Jacobian the identity matrix?
flat, unflatten = jax.flatten_util.ravel_pytree((v, A))


def algorithm_flat(f):
    return jax.flatten_util.ravel_pytree(algorithm(*unflatten(f)))[0]


fx, vjp = jax.vjp(algorithm_flat, flat)

# e1 = jnp.eye(len(fx))[0]
# e1 = jnp.arange(1.0, len(fx) + 1.0)
# _ = vjp(e1)
# assert False
# print(algorithm_flat(flat))


assert jnp.allclose(flat, algorithm_flat(flat))


# Flattened Jacobian: is it the identity matrix?
jac = jax.jacrev(algorithm_flat)(flat)


print(jac)


# print(jax.jacrev(algorithm, argnums=(0, 1))(v, A))
# assert False
