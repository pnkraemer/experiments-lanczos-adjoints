"""Test the tri-diagonalisation."""

import os

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import test_util
from tueplots import figsizes, fontsizes

from matfree_extensions import exp_util, lanczos

plt.rcParams.update(figsizes.neurips2023(nrows=1, ncols=2))
plt.rcParams.update(fontsizes.neurips2023(default_smaller=2))

jnp.set_printoptions(3)


def random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    flat_like = jnp.arange(1.0, 1.0 + len(flat))
    unflat = unflatten(flat_like)
    return jax.tree_util.tree_map(lambda s: s / jnp.mean(s), unflat)


def _sym(m):
    return jnp.triu(m) - jnp.diag(0.5 * jnp.diag(m))


# Set up a test-matrix
n = 50
eigvals = jnp.arange(2.0, 2.0 + n)
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = _sym(matrix)

# Set up an initial vector
vector = jnp.arange(4.0, 4.0 + n)


def matvec(v, p):
    return (p + p.T) @ v


krylov_depth = 3 * n // 4
(xs, (alphas, betas)), (x, beta) = lanczos.forward(
    matvec, krylov_depth, vector, params, reortho=False
)


(dxs, (dalphas, dbetas)), (dx, dbeta) = random_like((xs, (alphas, betas)), (x, beta))

xs = jnp.concatenate([xs, x[None]])
betas = jnp.concatenate([betas, beta[None]])
dxs = jnp.concatenate([dxs, dx[None]])
dbetas = jnp.concatenate([dbetas, dbeta[None]])

# Important: reorthogonalisation only works for dxs=0 for now...
dxs = jnp.zeros_like(dxs)


fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

for reortho, axis in zip([True, False], axes):
    gradients, (lambda_0, lambda_1N, mus, nus) = lanczos.adjoint(
        matvec=matvec,
        params=(params,),
        initvec_norm=jnp.linalg.norm(vector),
        alphas=alphas,
        betas=betas,
        xs=xs,
        dalphas=dalphas,
        dbetas=dbetas,
        dxs=dxs,
        reortho=reortho,
    )
    lambdas = jnp.concatenate([lambda_0[None], lambda_1N])
    orthogonality = lambdas @ xs.T
    plot_vals = jnp.log10(jnp.abs(orthogonality))

    color_values = axis.imshow(plot_vals, cmap="RdBu_r", vmin=-4, vmax=4)

    axis.set_title(f"Reortho: {str(reortho)}")
    axis.set_xlabel(r"State $x_k$")
    axis.set_ylabel(r"Adjoint $\lambda_k$")

label = r"$\log_{10}|\langle \lambda,  x\rangle|$"
fig.colorbar(color_values, ax=axes, orientation="vertical", pad=0.2, label=label)


directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}/figure.pdf")
plt.show()
