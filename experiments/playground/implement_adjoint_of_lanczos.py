"""Implement the adjoint of the Lanczos process."""
import jax
import jax.numpy as jnp
from matfree import test_util


def lanczos_init(matrix, vec):
    """Initialize Lanczos' algorithm.

    Solve A x_{k} = a_k x_k + b_k x_{k+1}
    for x_{k+1}, a_k, and b_k, using
    orthogonality of the x_k.
    """
    a = vec @ (matrix @ vec)
    r = matrix @ vec - a * vec
    b = jnp.linalg.norm(r)
    x = r / jnp.linalg.norm(b)
    return (x, b), a


def lanczos_step(matrix, vec, b, vec_previous):
    """Apply Lanczos' recurrence.

    Solve
    A x_{k} = b_{k-1} x_{k-1} + a_k x_k + b_k x_{k+1}
    for x_{k+1}, a_k, and b_k, using
    orthogonality of the x_k.
    """
    a = vec @ (matrix @ vec)
    r = matrix @ vec - a * vec - b * vec_previous
    b = jnp.linalg.norm(r)
    x = r / jnp.linalg.norm(b)
    return (x, b), a


eigvals = jnp.arange(1.0, 2.0, step=0.2)
A = test_util.symmetric_matrix_from_eigenvalues(eigvals)

key = jax.random.PRNGKey(1)
v0 = jax.random.normal(key, shape=jnp.shape(eigvals))
v0 /= jnp.linalg.norm(v0)

(v1, offdiag), diag = lanczos_init(A, v0)

while offdiag > 100 * jnp.finfo(jnp.dtype(offdiag)).eps:
    ((v1, offdiag), diag), v0 = lanczos_step(A, v1, offdiag, v0), v1
    print(offdiag)
