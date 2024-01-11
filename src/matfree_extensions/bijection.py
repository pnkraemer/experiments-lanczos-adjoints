"""Bijections: parametrised, invertible functions."""
from typing import Callable

import jax.numpy as jnp

REGISTRY_INVERSES: dict[Callable, Callable] = {}


def invert(func, /):
    try:
        return REGISTRY_INVERSES[func]
    except KeyError:
        raise KeyError("Function not registered.")


def linear(A, /):
    def func(x, /):
        return A @ x

    def func_inv(y, /):
        return jnp.linalg.solve(A, y)

    _register(func, func_inv)
    return func


def _register(func, func_inv, /):
    global REGISTRY_INVERSES
    REGISTRY_INVERSES[func] = func_inv
    REGISTRY_INVERSES[func_inv] = func


#
# def affine(x, /, params):
#     matvec, matvec_params, b = params
#     return matvec(x, *matvec_params) + b
#
#
# def _affine_inverse(y, /, params):
#     matvec, matvec_params, b = params
#
#     zeros = jnp.zeros_like(b)
#
#     def matvec_new(s, *p):
#         def matvec_p(z):
#             return matvec(z, *p)
#
#         return jax.scipy.linalg.gmres(matvec_p, s - b)
#
#     return affine(y, (matvec_new, matvec_params, zeros))
#
#
# REGISTRY_INVERSES[affine] = _affine_inverse
# REGISTRY_INVERSES[_affine_inverse] = affine
