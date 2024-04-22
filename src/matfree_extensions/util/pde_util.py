"""Partial differential equation utilities."""

from typing import Callable

import jax
import jax.numpy as jnp

from matfree_extensions import arnoldi


def mesh_2d_tensorproduct(x, y, /):
    return jnp.stack(jnp.meshgrid(x, y))


def stencil_2d_laplacian(dx):
    stencil = jnp.asarray([[0.0, -1.0, 0.0], [-1, 2.0, -1], [0.0, -1.0, 0.0]])
    return stencil / dx**2


def solution_terminal(*, init, rhs, expm):
    # todo: make "how to compute the dense expm" an argument
    def parametrize(p_init, p_rhs):
        def operator(t, x):
            y0 = init(**p_init)(x)
            y0_flat, unflatten = jax.flatten_util.ravel_pytree(y0)

            def matvec_p(v, p):
                Av = rhs(**p)(unflatten(v))
                return jax.flatten_util.ravel_pytree(Av)[0]

            algorithm = expm(matvec_p)

            Q, H, _r, c = algorithm(y0_flat, p_rhs)
            e1 = jnp.eye(len(H))[0, :]

            H = (H + H.T) / 2
            eigvals, eigvecs = jnp.linalg.eigh(H)

            expmat = eigvecs @ jnp.diag(jnp.exp(t * eigvals)) @ eigvecs.T
            # expmat = jax.scipy.linalg.expm(t * H)
            return unflatten(c * Q @ expmat @ e1)

        return operator

    return parametrize


# todo: other initial conditions


def pde_2d_init_bell():
    def parametrize(*, center):
        def fun(x, /):
            assert x.ndim == 3, jnp.shape(x)
            assert x.shape[0] == 2

            diff = x - center[:, None, None]

            def bell(d):
                # todo: make the "50" a parameter?
                return jnp.exp(-50 * jnp.dot(d, d))

            bell = jax.vmap(bell, in_axes=-1, out_axes=-1)
            bell = jax.vmap(bell, in_axes=-1, out_axes=-1)
            return bell(diff)

        return fun

    params = {"center": jnp.empty(())}
    return parametrize, params


def boundary_dirichlet():
    def pad(x, /):
        return jnp.pad(x, 1, mode="constant", constant_values=0.0)

    return pad


def boundary_neumann():
    def pad(x, /):
        return jnp.pad(x, 1, mode="edge")

    return pad


# todo: other rhs (e.g. Laplace + NN drift)?
def pde_2d_rhs_laplacian(*, stencil, boundary: Callable):
    def parametrize(*, intensity_sqrt):
        # todo: remove the stop_gradient
        intensity_sqrt = jax.lax.stop_gradient(intensity_sqrt)

        def rhs(x, /):
            assert x.ndim == 2, jnp.shape(x)
            assert x.shape[0] == x.shape[-1]

            x_padded = boundary(x)
            fx = jax.scipy.signal.convolve2d(stencil, x_padded, mode="valid")
            fx *= -(intensity_sqrt**2)  # todo: other positivity transforms?
            return fx

        return rhs

    return parametrize, {"intensity_sqrt": jnp.empty(())}


def loss_mse():
    def loss(sol, /, *, targets):
        return jnp.sqrt(jnp.mean((sol - targets) ** 2))

    return loss


def expm_arnoldi(krylov_depth, *, reortho="full", custom_vjp=True):
    def expm(matvec):
        return arnoldi.hessenberg(
            matvec, krylov_depth, reortho=reortho, custom_vjp=custom_vjp
        )

    return expm
