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


def shift(b, /):
    def func(x, /):
        return x + b

    def func_inv(y, /):
        return y - b

    _register(func, func_inv)
    return func


def elwise_tanh():
    _register(jnp.tanh, jnp.arctanh)
    return jnp.tanh


def _register(func, func_inv, /):
    global REGISTRY_INVERSES
    REGISTRY_INVERSES[func] = func_inv
    REGISTRY_INVERSES[func_inv] = func
