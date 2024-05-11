"""Conjugate gradient solvers."""

from typing import Callable

import jax
import jax.numpy as jnp


def cg_fixed_step(num_matvecs: int, /):
    pcg_solve = pcg_fixed_step(num_matvecs)

    def cg(A: Callable, b: jax.Array):
        return pcg_solve(A, b, lambda v: v)

    return cg


def pcg_fixed_step(num_matvecs: int, /):
    def pcg(A: Callable, b: jax.Array, P: Callable):
        return jax.lax.custom_linear_solve(
            A, b, lambda a, r: pcg_impl(a, r, P=P), symmetric=True, has_aux=True
        )

    def pcg_impl(A: Callable, b, P):
        x = jnp.zeros_like(b)

        r = b - A(x)
        z = P(r)
        p = z

        body_fun = make_body(A, P)
        init = (x, p, r, z)
        x, p, r, z = jax.lax.fori_loop(0, num_matvecs, body_fun, init_val=init)
        return x, {"residual": r}

    def make_body(A, P):
        def body_fun(_i, state):
            x, p, r, z = state

            Ap = A(p)
            a = jnp.dot(r, z) / (p.T @ Ap)
            x = x + a * p

            rold = r
            r = r - a * Ap

            zold = z
            z = P(r)
            b = jnp.dot(r, z) / jnp.dot(rold, zold)
            p = z + b * p
            return x, p, r, z

        return body_fun

    return pcg


def cg_fixed_step_reortho(num_matvecs: int, /):
    pcg_solve = pcg_fixed_step_reortho(num_matvecs)

    def cg(A: Callable, b: jax.Array):
        return pcg_solve(A, b, lambda v: v)

    return cg


def pcg_fixed_step_reortho(num_matvecs: int, /):
    def pcg(A: Callable, b: jax.Array, P: Callable):
        return jax.lax.custom_linear_solve(
            A, b, lambda a, r: pcg_impl(a, r, P=P), symmetric=True, has_aux=True
        )

    def pcg_impl(A: Callable, b, P):
        x = jnp.zeros_like(b)

        r = b - A(x)
        z = P(r)
        p = z

        Q = jnp.zeros((len(b), num_matvecs))
        step = pcg_step(A, P)
        init = (Q, x, p, r, z, jnp.dot(r, z))
        Q, x, _p, r, _z, _rzdot = jax.lax.fori_loop(0, num_matvecs, step, init)
        return x, {"residual": r, "Q": Q}

    def pcg_step(A, P):
        def body_fun(i, state):
            Q, x, p, r, z, rzdot = state

            # Start as usual
            Ap = A(p)
            a = _div(rzdot, (p.T @ Ap))
            x = x + a * p

            # Update
            r, rold = r - a * Ap, r
            z, zold = P(r), z

            # Reorthogonalise (don't forget to reassign z!)
            Q = Q.at[:, i].set(_div(rold, jnp.sqrt(rzdot)))
            r = r - Q @ (Q.T @ z)
            z = P(r)

            # Complete the step
            rzdot = jnp.dot(r, z)
            b = _div(rzdot, jnp.dot(rold, zold))
            p = z + b * p
            return Q, x, p, r, z, rzdot

        def _div(a, b, /):
            # Save division, so that the iteration can
            # run beyond convergence without dividing by zero
            eps = jnp.finfo(a.dtype).eps ** 2

            # Pre-clip to make all paths in where()
            #  NaN-free. See:
            #  https://github.com/google/jax/issues/5039
            b_safe = jnp.where(jnp.abs(b) > eps, b, 1.0)
            return jnp.where(jnp.abs(b) > eps, a / b_safe, a)

        return body_fun

    return pcg
