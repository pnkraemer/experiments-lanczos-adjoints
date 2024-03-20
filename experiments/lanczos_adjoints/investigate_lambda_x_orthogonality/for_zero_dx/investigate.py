"""Verify that for zero dx, reorthogonalisation works as promised."""

import os

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matfree import test_util
from matfree_extensions import exp_util, lanczos
from tueplots import figsizes, fontsizes

plt.rcParams.update(figsizes.neurips2023(nrows=2, ncols=2, rel_width=1.0))
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
n = 30
eigvals = jnp.arange(2.0, 2.0 + n)
matrix = test_util.symmetric_matrix_from_eigenvalues(eigvals)
params = _sym(matrix)

# Set up an initial vector
vector = jnp.arange(4.0, 4.0 + n)


def matvec(v, p):
    return (p + p.T) @ v


# krylov_depth = 3 * n // 4
krylov_depth = n - 1

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, dpi=200)

for reortho, axis in zip([True, False], axes):
    (xs, (alphas, betas)), (x, beta) = lanczos.forward(
        matvec, krylov_depth, vector, params, reortho=True
    )

    (dxs, (dalphas, dbetas)), (dx, dbeta) = random_like(
        (xs, (alphas, betas)), (x, beta)
    )

    xs = jnp.concatenate([xs, x[None]])
    betas = jnp.concatenate([betas, beta[None]])
    dxs = jnp.concatenate([dxs, dx[None]])
    dbetas = jnp.concatenate([dbetas, dbeta[None]])

    # Important: reorthogonalisation only works for dxs=0 for now...
    dxs = jnp.zeros_like(dxs)
    gradients, (lambda_0, lambda_1N, mus, nus, _xis) = lanczos.adjoint(
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

    # A true zero would break this plotting code
    # (-inf is sometimes plotted as zero),
    # so we clip with the machine epsilon
    eps = jnp.finfo(jnp.dtype(lambdas)).eps
    plot_vals = jnp.log10(eps + jnp.abs(orthogonality))

    color_values = axis.imshow(plot_vals, vmin=-7, vmax=1, interpolation="none")

    axis.set_title(f"Reortho: {reortho!s}")
    axis.set_xlabel(r"State index $x_k$")

axes[0].set_ylabel(r"Adjoint index $\lambda_k$")

label = r"$\log_{10}|\langle \lambda,  x\rangle|$"
fig.colorbar(
    color_values, ax=axes, orientation="vertical", pad=0.1, label=label, shrink=0.6
)


directory = exp_util.matching_directory(__file__, "figures/")
os.makedirs(directory, exist_ok=True)

plt.savefig(f"{directory}/figure.pdf")
plt.show()
