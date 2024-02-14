"""Explore."""

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import test_util
from matfree_extensions import lanczos


def random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    flat_like = jnp.arange(1.0, 1.0 + len(flat), step=1.0)
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)


# def _sym(m):
#     return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


jnp.set_printoptions(5, suppress=True)

# Set up a test-matrix
n = 15
eigvals = jnp.arange(2.0, 2.0 + n)
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = matrix


def matvec(v, p):
    return p @ v


# Set up an initial vector
vector = jnp.arange(4.0, 4.0 + n)
vector /= jnp.linalg.norm(vector)


# krylov_depth = 3 * n // 4
# krylov_depth = n // 2 + 1
# krylov_depth = n // 2
krylov_depth = n // 2
# krylov_depth=1


# Forward pass
def fwd(v0, p):
    Q, (A, B), R, norm = lanczos.matrix_forward(matvec, krylov_depth, v0, p)
    return Q, (A, B), R, norm


(xs, (alphas, betas), residual, length), VJP = jax.vjp(fwd, vector, params)

# Random gradient values to backward-pass
(dxs, (dalphas, dbetas), dresidual, dlength) = random_like(xs, (alphas, betas), residual, length)
dv_ref, dp_ref = VJP((dxs, (dalphas, dbetas), dresidual, dlength))


# Adjoint pass (to be computed)
(dv2, dp2), (_rho, L, M, Xi) = lanczos.matrix_adjoint(
    matvec=matvec,
    params=(params,),
    norm=length,
    alphas=alphas,
    betas=betas,
    xs=xs,
    res=residual,
    dalphas=dalphas,
    dbetas=dbetas,
    dxs=dxs,
    dnorm=dlength,
    dres=dresidual,
)


# Print gradients wrt v0.
#
# todo: for full-rank Lanczos approximations,
#  the matrix-adjoint yields a zero gradient wrt v
#  which feels like the correct answer!
#  Autodiff does not yield this zero-gradient. Why?
#  does the matrix adjoint compute the adjoint of a different decomposition?
#  Also, dv2 is orthogonal to _all_ Xs, not just the first one.
#  what does that mean?
#

print(dv_ref @ xs.T)
print()
print(dv1 @ xs.T)
print()
print()
print(dv2 @ xs.T)
print()


# Plot the bi-orthogonality of lambda and X
ortho = L @ xs.T
ortho = jnp.log10(jnp.abs(ortho) + jnp.finfo(jnp.dtype(ortho)).eps)
plt.title("Log(L.T @ X)")
plt.imshow(ortho)
plt.colorbar()
plt.show()
