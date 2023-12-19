"""Extensions for the Matfree package."""

from matfree import decomp
from matfree.backend import func, linalg, tree_util


def integrand_slq_spd_with_grad(matfun, order, matvec, /):
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

        def matvec_flat_p(v_flat, *p):
            v = v_unflatten(v_flat)
            Av = matvec(v, *p)
            flat, unflatten = tree_util.ravel_pytree(Av)
            return flat

        _, vjp = func.vjp(lambda *p: matvec_flat_p(w2, *p).T @ w1, *parameters)
        grad = vjp(1.0)

        # Return both
        return {"value": slqval, "grad": grad}

    return quadform
