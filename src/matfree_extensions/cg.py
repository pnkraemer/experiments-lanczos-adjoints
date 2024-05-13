"""Conjugate gradient solvers."""

from typing import Callable

import jax
import jax.numpy as jnp


def cg_fixed_step(*args, **kwargs):
    pcg_solve = pcg_fixed_step(*args, **kwargs)

    def cg(A: Callable, b: jax.Array):
        return pcg_solve(A, b, lambda v: v)

    return cg


def pcg_fixed_step(num_matvecs: int, /):
    def pcg(A: Callable, b: jax.Array, P: Callable):
        # Uncomment if we want to print values from inside the solver:
        # return pcg_impl(A, b, P=P)

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
        return x, {"residual_abs": r, "residual_rel": r / jnp.abs(x)}

    def make_body(A, P):
        def body_fun(_i, state):
            x, p, r, z = state

            Ap = A(p)
            a = _safe_divide(jnp.dot(r, z), (p.T @ Ap))
            x = x + a * p

            rold = r
            r = r - a * Ap

            zold = z
            z = P(r)

            b = _safe_divide(jnp.dot(r, z), jnp.dot(rold, zold))
            p = z + b * p
            return x, p, r, z

        return body_fun

    return pcg


def cg_adaptive(**kwargs):
    pcg_solve = pcg_adaptive(**kwargs)

    def cg(A: Callable, b: jax.Array):
        return pcg_solve(A, b, lambda v: v)

    return cg


# atol and rtol follow jnp.allclose logic
def pcg_adaptive(*, atol: float, rtol, maxiter: int, miniter: int):
    def pcg(A: Callable, b: jax.Array, P: Callable):
        # Uncomment if we want to print values from inside the solver:
        # return pcg_impl(A, b, P=P)

        return jax.lax.custom_linear_solve(
            A, b, lambda a, r: pcg_impl(a, r, P=P), symmetric=True, has_aux=True
        )

    def pcg_impl(A: Callable, b, P):
        x = jnp.zeros_like(b)

        r = b - A(x)
        z = P(r)
        p = z

        cond_fun = make_cond()
        body_fun = make_body(A, P)
        init = (x, p, r, z, 0.0)
        x, p, r, z, num_steps = jax.lax.while_loop(cond_fun, body_fun, init)
        return x, {
            "residual_abs": r,
            "residual_rel": r / jnp.abs(x),
            "num_steps": num_steps,
        }

    def make_cond():
        def cond(state):
            x, p, r, z, nsteps = state

            error_rel = r / (atol + jnp.abs(x) * rtol)
            is_error_large = jnp.sqrt(jnp.mean(error_rel**2)) > 1.0
            is_first_step = nsteps < miniter
            proceed = jnp.logical_or(is_error_large, is_first_step)
            is_not_beyond_maxiter = nsteps < maxiter
            return jnp.logical_and(proceed, is_not_beyond_maxiter)

        return cond

    def make_body(A, P):
        def body_fun(state):
            x, p, r, z, nsteps = state

            Ap = A(p)
            a = _safe_divide(jnp.dot(r, z), (p.T @ Ap))
            x = x + a * p

            rold = r
            r = r - a * Ap

            zold = z
            z = P(r)

            b = _safe_divide(jnp.dot(r, z), jnp.dot(rold, zold))
            p = z + b * p
            return x, p, r, z, nsteps + 1

        return body_fun

    return pcg


def cg_fixed_step_reortho(*args, **kwargs):
    pcg_solve = pcg_fixed_step_reortho(*args, **kwargs)

    def cg(A: Callable, b: jax.Array):
        return pcg_solve(A, b, lambda v: v)

    return cg


def pcg_fixed_step_reortho(num_matvecs: int, /):
    def pcg(A: Callable, b: jax.Array, P: Callable):
        # Uncomment if we want to print values from inside the solver:
        # return pcg_impl(A, b, P=P)

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
        return x, {"residual_abs": r, "Q": Q}

    def pcg_step(A, P):
        def body_fun(i, state):
            Q, x, p, r, z, rzdot = state

            # Start as usual
            Ap = A(p)
            # print(p.T @ Ap)

            a = _safe_divide(rzdot, (p.T @ Ap))
            x = x + a * p

            # Update
            r, rold = r - a * Ap, r
            z, zold = P(r), z

            # Reorthogonalise (don't forget to reassign z!)
            Q = Q.at[:, i].set(_safe_divide(rold, _safe_sqrt(rzdot)))
            # print("ORTHO", Q.T @ z)
            r = r - Q @ (Q.T @ z)
            z = P(r)
            # print("ORTHO", Q.T @ z)
            # print()

            # Complete the step
            rzdot = jnp.dot(r, z)
            b = _safe_divide(rzdot, jnp.dot(rold, zold))
            p = z + b * p
            return Q, x, p, r, z, rzdot

        return body_fun

    return pcg


def _safe_divide(a, b, /):
    # Safe division, so that the iteration can
    # run beyond convergence without dividing by zero.
    # This happens in CG when the iteration has converged,
    # in which case we would compute 0/0. By using this
    # safe_divide, we avoid NaNs.
    # See, e.g.
    # https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/linear_cg.py

    # Pre-clip to make all paths in where()
    #  NaN-free. See:
    #  https://github.com/google/jax/issues/5039
    eps = jnp.finfo(a.dtype).eps ** 2
    b_safe = jnp.where(jnp.abs(b) > eps, b, 1.0)
    return jnp.where(jnp.abs(b) > eps, a / b_safe, a)


def _safe_sqrt(a, /):
    a_safe = jnp.where(a > 0.0, a, 0.0)
    return jnp.sqrt(a_safe)
