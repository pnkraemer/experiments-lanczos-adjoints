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
        decompose = lanczos_fwd(matvec, depth=len(vec) - 1)

        # Lanczos-decompose
        (vecs, diags, offdiags) = decompose(vec, matrix)
        v0 = vecs[0]

        # Reconstruct the original matrix
        # (if full-rank Lanczos, this should imply an identity-Jacobian)
        dense_matrix = jnp.diag(diags) + jnp.diag(offdiags, 1) + jnp.diag(offdiags, -1)
        return v0, sym(vecs.T @ dense_matrix @ vecs)

    return identity


def lanczos_fwd(matvec, depth):
    def normal(s):
        return

    def estimate(vec, *params):
        k, beta, q, r = 0, 1, jnp.zeros_like(vec), vec

        vs = jnp.zeros((depth + 1, len(vec)))
        diags = jnp.zeros((depth + 1,))
        offdiags = jnp.zeros((depth + 1,))
        for i in range(depth + 1):
            q, q0 = r / beta, q

            q = q - vs.T @ (vs @ q)
            q = (
                q
                - jax.lax.stop_gradient(q)
                + jax.lax.stop_gradient(q / jnp.linalg.norm(q))
            )

            k = k + 1
            Aq = matvec(q, *params)
            alpha = q @ Aq
            r = Aq - alpha * q - beta * q0
            beta = jnp.linalg.norm(r)

            vs = vs.at[i].set(q)
            diags = diags.at[i].set(alpha)
            offdiags = offdiags.at[i].set(beta)

        return vs, diags, offdiags[:-1]

    return estimate


# Set up a test-matrix
eigvals = jnp.ones((5,), dtype=float) * 0.001
eigvals_relevant = jnp.arange(1.0, 2.0, step=1.0 / len(eigvals))
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


print("flattened input:\n\t", flat)
print("flattened output:\n\t", algorithm_flat(flat))
assert jnp.allclose(flat, algorithm_flat(flat))


# Flattened Jacobian: is it the identity matrix?
# Only works when run as python -O experiments/....py
# because the asserts fail.
jac = jax.jacfwd(algorithm_flat)(flat)

plt.imshow(remove_zero_rows(jac))
plt.colorbar()
plt.show()


jnp.set_printoptions(2)
X = jac[: len(v), : len(v)]
print("top-left block:\n\t", X)
print("eigvals:\n\t", jnp.linalg.eigvalsh(X))


assert False


@jax.custom_vjp
def normalize(vec):
    return vec / jnp.linalg.norm(vec)


def normalize_fwd(vec):
    x = normalize(vec)
    normm = jnp.linalg.norm(vec)
    return x, (x, normm)


def normalize_bwd(cache, vjp_incoming):
    x, normm = cache
    return (vjp_incoming / normm - (vjp_incoming @ x) * x / normm**3,)


normalize.defvjp(normalize_fwd, normalize_bwd)

print()

print(v)
print(normalize(v))

print(jax.jacrev(normalize)(v))
print()
print(jnp.eye(len(v)) - jnp.outer(v, v))
