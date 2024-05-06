"""Partial differential equation utilities."""

import functools
from typing import Callable, Sequence

import diffrax
import flax.linen
import jax
import jax.numpy as jnp

from matfree_extensions import arnoldi


def mesh_tensorproduct(x, y, /):
    return jnp.stack(jnp.meshgrid(x, y))


def stencil_laplacian(dx):
    stencil = jnp.asarray([[0.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, 0.0]])
    return stencil / dx**2


def stencil_advection_diffusion(dx):
    diffusion = jnp.asarray([[0.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, 0.0]])
    diffusion = diffusion / dx**2
    advection = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
    advection = advection / (2 * dx)
    return diffusion + advection


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


def pde_heat_anisotropic(scale_like, /, stencil, *, constrain, boundary: Callable):
    def parametrize(*, scale):
        scale_constrained = constrain(scale)  # e.g. ensure positivity

        def rhs(x, /):
            assert x.ndim == 3, jnp.shape(x)
            assert x.shape[1] == x.shape[2]
            assert x.shape[0] == 2

            u, du = x
            x_padded = boundary(u)
            fx = jax.scipy.signal.convolve2d(stencil, x_padded, mode="valid")
            u_new = -fx * scale_constrained
            return jnp.stack([u_new, du])

        return rhs

    return parametrize, {"scale": jnp.empty_like(scale_like)}


def pde_wave_anisotropic(scale_like, /, stencil, *, constrain, boundary: Callable):
    def parametrize(*, scale):
        scale_constrained = constrain(scale)  # e.g. ensure positivity

        def rhs(x, /):
            assert x.ndim == 3, jnp.shape(x)
            assert x.shape[1] == x.shape[2]
            assert x.shape[0] == 2

            u, du = x
            x_padded = boundary(u)
            fx = jax.scipy.signal.convolve2d(stencil, x_padded, mode="valid")
            u_new = fx * scale_constrained
            return jnp.stack([du, u_new])

        return rhs

    return parametrize, {"scale": jnp.empty_like(scale_like)}


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
        return jnp.mean((sol - targets) ** 2)

    return loss

def loss_rmse():
    def loss(sol, /, *, targets):
        nugget = jnp.sqrt(jnp.finfo(targets).eps)
        return jnp.mean((sol - targets) ** 2 / (nugget+jnp.abs(targets)))

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
        (_t1, y1), _y_all = jax.lax.scan(step, xs=dts, init=(t0, y0))
        return y1

    return solve


def solver_diffrax(
    t0, t1, vector_field, /, *, num_steps: int, method: str, adjoint: str
):
    @diffrax.ODETerm
    def term(t, y, args):  # noqa: ARG001
        return vector_field(y, args)

    match_methods = {
        "dopri5": diffrax.Dopri5(),
        "tsit5": diffrax.Tsit5(),
        "euler": diffrax.Euler(),
        "heun": diffrax.Heun(),
    }
    solver = match_methods[method]

    match_adjoints = {
        "recursive_checkpoint": diffrax.RecursiveCheckpointAdjoint(),
        "direct": diffrax.DirectAdjoint(),
        "backsolve": diffrax.BacksolveAdjoint(),
    }
    backprop = match_adjoints[adjoint]

    dt0 = (t1 - t0) / num_steps
    stepsize_controller = diffrax.ConstantStepSize()

    def solve(y0, p):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            args=p,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            stepsize_controller=stepsize_controller,
            adjoint=backprop,
        )
        return sol.ys[-1]

    return solve


def solver_expm(t0, t1, vector_field, /, expm):
    # todo: turn into a solver
    # todo: make "how to compute the dense expm" an argument
    def solve(y0, *p):
        y0_flat, unflatten_y = jax.flatten_util.ravel_pytree(y0)

        def matvec_p(v, p_):
            Av = vector_field(unflatten_y(v), *p_)
            return jax.flatten_util.ravel_pytree(Av)[0]

        dt = t1 - t0
        expm_matvec = expm(matvec_p, dt, y0_flat, p)
        return unflatten_y(expm_matvec)

    return solve


def expm_arnoldi(
    krylov_depth, *, max_squarings: int = 32, reortho="full", custom_vjp=True
):
    def expm(matvec, dt, y0_flat, *p):
        kwargs = {"reortho": reortho, "custom_vjp": custom_vjp}
        algorithm = arnoldi.hessenberg(matvec, krylov_depth, **kwargs)
        Q, H, _r, c = algorithm(y0_flat, *p)
        e1 = jnp.eye(len(H))[0, :]
        expmat = jax.scipy.linalg.expm(dt * H, max_squarings=max_squarings)
        return 1 / c * Q @ expmat @ e1

    return expm


def expm_pade():
    def expm(matvec, dt, y0_flat, *p):
        # Materialise the matrix
        matrix = jax.jacfwd(lambda v: matvec(v, *p))(y0_flat)

        # Compute the matrix exponential
        return jax.scipy.linalg.expm(dt * matrix) @ y0_flat

    return expm


def model_mlp(mesh_like, features, /, activation: Callable, *, output_scale_raw: float):
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

    # We scale the outputs down to not accidentally
    #  initialise a too-large-in-magnitude PDE parameter
    #  which would then lead to solutions blowing up.
    output_scale = _softplus(output_scale_raw)

    def apply(params, args):
        # Reshape into Flax's desired format
        args_ = args.reshape((2, -1)).T

        # Apply and shape back into our desired format
        fx = model.apply(params, args_).reshape((-1,))
        fx *= output_scale
        return fx.reshape(args[0].shape)

    return init, apply


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
