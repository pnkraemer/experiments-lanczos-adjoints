"""Bijections: parametrised, invertible functions."""
from typing import Callable

import jax.numpy as jnp

_REGISTRY_INVERSES: dict[Callable, Callable] = {}


def chain(func1, func2, /):
    def func(x, /):
        return func2(func1(x))

    def func_inv(y, /):
        return invert(func1)(invert(func2)(y))

    _register(func, func_inv)
    return func


def invert(func, /):
    try:
        return _REGISTRY_INVERSES[func]
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
    global _REGISTRY_INVERSES
    _REGISTRY_INVERSES[func] = func_inv
    _REGISTRY_INVERSES[func_inv] = func
