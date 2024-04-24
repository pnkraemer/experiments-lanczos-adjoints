"""Partial differential equation utilities."""

import functools
from typing import Callable, Sequence

import flax.linen
import jax
import jax.numpy as jnp

from matfree_extensions import arnoldi


# todo: write code for discretisation, and code for solving ODEs
#  build different solvers (the one-step thing below does not work)
#  start small (by comparing euler to chunk-euler+lanczos), and then slowly
#  move to crank-nicholson
#  maybe even start by implementing euler or even diffrax version or whatnot?
#  to get a feeling for api and then build the cool arnoldi-solvers
def mesh_tensorproduct(x, y, /):
    return jnp.stack(jnp.meshgrid(x, y))


def stencil_laplacian(dx):
    stencil = jnp.asarray([[0.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, 0.0]])
    return stencil / dx**2


def solution_terminal(*, init, rhs, expm):
    # todo: turn into a solver
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


def expm_arnoldi(krylov_depth, *, reortho="full", custom_vjp=True):
    def expm(matvec):
        return arnoldi.hessenberg(
            matvec, krylov_depth, reortho=reortho, custom_vjp=custom_vjp
        )

    return expm


# todo: other initial conditions


def pde_init_bell(c, /):
    def parametrize(*, center_logits):
        center = _sigmoid(center_logits)

        def fun(x, /):
            assert x.ndim == 3, jnp.shape(x)
            assert x.shape[0] == 2

            diff = x - center[:, None, None]

            def bell(d):
                return jnp.exp(-(c**2) * jnp.dot(d, d))

            bell = jax.vmap(bell, in_axes=-1, out_axes=-1)
            bell = jax.vmap(bell, in_axes=-1, out_axes=-1)
            return bell(diff)

        return fun

    params = {"center_logits": jnp.empty((2,))}
    return parametrize, params


def pde_init_sine():
    def parametrize(*, scale_sin, scale_cos):
        def fun(x, /):
            assert x.ndim == 3, jnp.shape(x)
            assert x.shape[0] == 2

            term_sin = jnp.sin(scale_sin * x[0])
            term_cos = jnp.cos(scale_cos * x[1])
            return term_sin * term_cos

        return fun

    params = {"scale_sin": 5.0, "scale_cos": 3.0}
    return parametrize, params


def _sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# todo: other rhs (e.g. Laplace + NN drift)?
def pde_heat(c: float, /, stencil, *, boundary: Callable):
    def parametrize():
        def rhs(x, /):
            assert x.ndim == 2, jnp.shape(x)
            assert x.shape[0] == x.shape[-1]

            x_padded = boundary(x)
            fx = jax.scipy.signal.convolve2d(stencil, x_padded, mode="valid")
            fx *= c
            return fx

        return rhs

    return parametrize, {}


def pde_heat_affine(c: float, drift_like, /, stencil, *, boundary: Callable):
    def parametrize(*, drift):
        def rhs(x, /):
            assert x.ndim == 2, jnp.shape(x)
            assert x.shape[0] == x.shape[-1]

            x_padded = boundary(x)
            fx = jax.scipy.signal.convolve2d(stencil, x_padded, mode="valid")
            fx *= c
            return fx + drift

        return rhs

    return parametrize, {"drift": jnp.empty_like(drift_like)}


def _softplus(x, beta=1.0, threshold=20.0):
    # Shamelessly stolen from:
    # https://github.com/google/jax/issues/18443

    # mirroring the pytorch implementation
    #  https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    x_safe = jax.lax.select(x * beta < threshold, x, jax.numpy.ones_like(x))
    return jax.lax.select(
        x * beta < threshold,
        1 / beta * jax.numpy.log(1 + jax.numpy.exp(beta * x_safe)),
        x,
    )


def boundary_dirichlet():
    def pad(x, /):
        return jnp.pad(x, 1, mode="constant", constant_values=0.0)

    return pad


def boundary_neumann():
    def pad(x, /):
        return jnp.pad(x, 1, mode="edge")

    return pad


def loss_mse():
    def loss(sol, /, *, targets):
        return jnp.sqrt(jnp.mean((sol - targets) ** 2))

    return loss


# Below here is proper tested
def solver_euler_fixed_step(ts, vector_field, /):
    def step_fun(t_and_y, dt, p):
        t, y = t_and_y
        t = t + dt
        y = y + dt * vector_field(y, *p)
        return (t, y), y

    def solve(y0, *p):
        t0, dts = ts[0], jnp.diff(ts)
        step = functools.partial(step_fun, p=p)
        return jax.lax.scan(step, xs=dts, init=(t0, y0))

    return solve


def model_pde(*, unflatten, init, solve):
    unflatten_p, unflatten_x = unflatten

    def model(p, x):
        mesh = unflatten_x(x)
        p_init, p_solve = unflatten_p(p)
        y0 = init(mesh, p_init)
        y1, y_all = solve(y0, p_solve)
        return y1, y_all

    return model


def model_mlp(mesh_like, features, /, activation: Callable):
    assert features[-1] == 1

    class MLP(flax.linen.Module):
        features: Sequence[int]

        @flax.linen.compact
        def __call__(self, x):
            for feat in self.features[:-1]:
                x = flax.linen.Dense(feat)(x)
                x = activation(x)
            return flax.linen.Dense(self.features[-1])(x)

    assert mesh_like.ndim == 3
    mesh_like = mesh_like.reshape((2, -1)).T

    model = MLP(features)

    def init(key):
        variables = model.init(key, mesh_like)
        return jax.flatten_util.ravel_pytree(variables)

    def apply(params, args):
        args_ = args.reshape((2, -1)).T
        fx = model.apply(params, args_).reshape((-1,))
        return fx.reshape(args[0].shape)

    return init, apply
