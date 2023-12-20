"""Extensions for the Matfree package."""

import jax
from matfree import decomp
from matfree.backend import func, linalg, tree_util


def integrand_slq_spd_value_and_grad(matfun, order, matvec, /):
    def quadform(v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)

        def matvec_flat(v_flat):
            v = v_unflatten(v_flat)
            Av = matvec(v, *parameters)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat

        algorithm = decomp.lanczos_tridiag_full_reortho(order)
        basis, tridiag = decomp.decompose_fori_loop(
            v0_flat, matvec_flat, algorithm=algorithm
        )
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = linalg.diagonal_matrix(diag)
        offdiag1 = linalg.diagonal_matrix(off_diag, -1)
        offdiag2 = linalg.diagonal_matrix(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0_flat.shape

        # Evaluate the matrix-function
        fx_eigvals = func.vmap(matfun)(eigvals)
        slqval = dim * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

        # Evaluate the derivative
        dfx_eigvals = func.vmap(func.jacfwd(matfun))(eigvals)
        sol = eigvecs @ (dfx_eigvals * eigvecs[0, :].T)
        w1, w2 = linalg.vector_norm(v0) * (basis.T @ sol), v0

        @tree_util.partial_pytree
        def matvec_flat_p(v_flat, *p):
            v = v_unflatten(v_flat)
            av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(av)
            return flat

        grad = jax.grad(lambda *pa: matvec_flat_p(w2, *pa).T @ w1)(*parameters)

        return slqval, grad

    return quadform


def integrand_slq_spd_custom_vjp(matfun, order, matvec, /):
    @jax.custom_vjp
    def quadform(v0, *parameters):
        # This function shall only be meaningful inside a VJP
        raise RuntimeError

    def quadform_fwd(v0, *parameters):
        v0_flat, v_unflatten = tree_util.ravel_pytree(v0)

        def matvec_flat(v_flat):
            v = v_unflatten(v_flat)
            Av = matvec(v, *parameters)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat

        algorithm = decomp.lanczos_tridiag_full_reortho(order)
        basis, tridiag = decomp.decompose_fori_loop(
            v0_flat, matvec_flat, algorithm=algorithm
        )
        (diag, off_diag) = tridiag

        # todo: once jax supports eigh_tridiagonal(eigvals_only=False),
        #  use it here. Until then: an eigen-decomposition of size (order + 1)
        #  does not hurt too much...
        diag = linalg.diagonal_matrix(diag)
        offdiag1 = linalg.diagonal_matrix(off_diag, -1)
        offdiag2 = linalg.diagonal_matrix(off_diag, 1)
        dense_matrix = diag + offdiag1 + offdiag2
        eigvals, eigvecs = linalg.eigh(dense_matrix)

        # Since Q orthogonal (orthonormal) to v0, Q v = Q[0],
        # and therefore (Q v)^T f(D) (Qv) = Q[0] * f(diag) * Q[0]
        (dim,) = v0_flat.shape

        # Evaluate the matrix-function
        fx_eigvals = func.vmap(matfun)(eigvals)
        slqval = dim * linalg.vecdot(eigvecs[0, :], fx_eigvals * eigvecs[0, :])

        # Evaluate the derivative
        dfx_eigvals = func.vmap(func.jacfwd(matfun))(eigvals)
        sol = eigvecs @ (dfx_eigvals * eigvecs[0, :].T)
        w1, w2 = linalg.vector_norm(v0) * (basis.T @ sol), v0

        @tree_util.partial_pytree
        def matvec_flat_p(v_flat, *p):
            v = v_unflatten(v_flat)
            av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(av)
            return flat

        # Return both
        cache = {
            "matvec_flat_p": matvec_flat_p,
            "w1": w1,
            "w2": w2,
            "parameters": parameters,
        }
        return slqval, cache

    def quadform_bwd(cache, vjp_incoming):
        fun = cache["matvec_flat_p"]
        p = cache["parameters"]
        w1, w2 = cache["w1"], cache["w2"]

        _fx, vjp = func.vjp(lambda *pa: fun(w2, *pa).T @ w1, *p)

        # todo: compute gradient wrt v?
        return 0.0, *vjp(vjp_incoming)

    quadform.defvjp(quadform_fwd, quadform_bwd)

    return quadform
