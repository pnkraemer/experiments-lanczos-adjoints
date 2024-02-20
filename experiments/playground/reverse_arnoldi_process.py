"""Invert a triangular matrix."""

import jax.numpy as jnp
from matfree_extensions import arnoldi, exp_util

# todo: merge this with the stability scripts to measure the reorthogonalisation-need over time.

for nrows in range(3, 33, 2):
    # Set up a test-matrix
    matrix = exp_util.hilbert(nrows)
    matrix = matrix + matrix.T

    def matvec(x, p):
        return p @ x

    # Set up an initial vector
    vector = jnp.ones((nrows,)) / jnp.sqrt(nrows)

    # prepare the algorithm
    krylov_depth = len(vector) // 2
    Q, H, r, c = arnoldi.forward(
        matvec, vector, krylov_depth, matrix, reortho="full_with_sparsity"
    )
    r1 = r
    r0 = Q[:, -1]

    offdiag, diag = jnp.diag(H, -1), jnp.diag(H)

    offdiag = jnp.concatenate((offdiag, jnp.ones((1,))), axis=0)

    Rt = jnp.zeros_like(Q.T)
    for i in range(len(H) - 1, -1, -1):
        alpha = diag[i]

        # print(Q[:, :i+1].T)
        # print(r0)
        #
        # print()
        # r0 = r0 - Q[:, :i] @ Q[:, :i].T @ r0
        # todo: orthogonalise against upcoming vectors to mimic
        #  "exploring the negative space" like in the adjoints
        r0 = r0 - Rt.T @ Rt @ r0
        r0 /= jnp.linalg.norm(r0)

        Rt = Rt.at[i, :].set(r0)

        # this is lanczos-specific...
        r00 = (-diag[i] * r0 + matrix @ r0 - offdiag[i] * r1) / offdiag[i - 1]

        # r00 = r00 - Rt.T @ Rt @ r00
        # print(jnp.linalg.norm(r00))

        r0, r1 = r00, r0

    print(
        nrows,
        jnp.linalg.norm(Rt @ Rt.T - jnp.eye(len(Rt))) / jnp.sqrt((Rt @ Rt.T).size),
    )
