"""Debug the implementation of the Lanczos-process."""

import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import lanczos, test_util


def sym(x):
    return jnp.triu(x) - jnp.diag(0.5 * jnp.diag(x))


def remove_zero_rows(m):
    row_is_zero = jnp.all(m == 0, axis=1)
    return m[jnp.logical_not(row_is_zero), :][:, jnp.logical_not(row_is_zero)]


# While debugging:
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)


def identity_via_lanczos(matvec, *, custom_vjp: bool):
    # Lanczos algorithm
    # decompose = lanczos_fwd(matvec, custom_vjp=custom_vjp)

    def identity(vec, matrix):
        # Use a matfree-reference
        decompose = lanczos.alg_tridiag_full_reortho(matvec, len(vec) - 1)

        # Lanczos-decompose
        (vecs, (diags, offdiags)) = decompose(vec, matrix)
        v0 = vecs[0]

        # Reconstruct the original matrix
        # (if full-rank Lanczos, this should imply an identity-Jacobian)
        dense_matrix = jnp.diag(diags) + jnp.diag(offdiags, 1) + jnp.diag(offdiags, -1)
        return v0, sym(vecs.T @ dense_matrix @ vecs)

    return identity


jnp.set_printoptions(1)

# Set up a test-matrix
eigvals = jnp.ones((3,), dtype=float) * 0.001
eigvals_relevant = jnp.arange(1.0, 2.0, step=1.0 / len(eigvals))[:3]
eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
A = test_util.symmetric_matrix_from_eigenvalues(eigvals)
P = sym(A)

# Set up an initial vector
v = jnp.arange(1.0, 1.0 + len(eigvals))
v /= jnp.linalg.norm(v)


# Verify that the "identity" operator is indeed correct:

# Flatten the algorithm to get a single Jacobian "matrix"!
flat, unflatten = jax.flatten_util.ravel_pytree((v, P))
algorithm = identity_via_lanczos(lambda s, p: (p + p.T) @ s, custom_vjp=False)


def algorithm_flat(f):
    return jax.flatten_util.ravel_pytree(algorithm(*unflatten(f)))[0]


assert jnp.allclose(flat, algorithm_flat(flat))


# Flattened Jacobian: is it the identity matrix?
# Only works when run as python -O experiments/....py
# because the asserts fail.
jac = jax.jacfwd(algorithm_flat)(flat)

plt.imshow(remove_zero_rows(jac))
plt.colorbar()
plt.show()
print(jac)
