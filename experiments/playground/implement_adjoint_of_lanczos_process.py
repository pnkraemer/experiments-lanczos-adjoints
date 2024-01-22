"""Implement the adjoint of the Lanczos process."""
import jax
import jax.numpy as jnp
from matfree import test_util

# todo:
#  (i) implement the lanczos decomposition such that it
#  returns the full decomposition plus all $a$s and $b$s
#  (ii) compute the VJP of this decomposition in a random direction
#  (iii) implement the custom VJP recursion
#  (Use the content of demonstrate_symmetry below)


assert False


def lanczos_fwd_init(matrix, vec):
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


def lanczos_fwd_step(matrix, vec, b, vec_previous):
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


def lanczos_rev_step(matrix, vec, b, vec_next):
    a = vec.T @ (matrix @ vec)
    r = (matrix @ vec) - a * vec - b * vec_next
    b = jnp.linalg.norm(r)
    x = r / b
    return (x, b), a


# Set up a test-matrix
eigvals = jnp.ones((50,), dtype=float) * 0.001
eigvals_relevant = jnp.arange(1.0, 2.0, step=0.1)
eigvals = eigvals.at[: len(eigvals_relevant)].set(eigvals_relevant)
A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

# Set up an initial vector
key = jax.random.PRNGKey(1)
v0 = jax.random.normal(key, shape=jnp.shape(eigvals))
v0 /= jnp.linalg.norm(v0)


# Initialize and apply the forward recursion
i, small_value = 0, jnp.sqrt(jnp.finfo(jnp.dtype(v0)).eps)
(v1, offdiag), diag = lanczos_fwd_init(A, v0)
while (offdiag > small_value) and (i := (i + 1)) < len(eigvals):
    ((v1, offdiag), diag), v0 = lanczos_fwd_step(A, v1, offdiag, v0), v1
    print(i, offdiag)

# Go backwards:
# Forward recursion, but change the roles of v0 and v1
# while (offdiag > small_value) and (i := (i + 1)) < len(eigvals):
for j in range(i - 1, 0, -1):
    ((v0, offdiag), diag), v1 = lanczos_fwd_step(A, v0, offdiag, v1), v0
    print(j, offdiag)
