import jax
import jax.numpy as jnp
import pytest
import pytest_cases

from matfree_extensions import bijection


@pytest_cases.case()
def case_linear():
    key = jax.random.PRNGKey(seed=4141231)
    A = jax.random.normal(key, shape=(3, 3))
    return bijection.linear(A)


@pytest_cases.case()
def case_shift():
    key = jax.random.PRNGKey(seed=6781974)
    b = jax.random.normal(key, shape=(3,))
    return bijection.shift(b)


@pytest_cases.parametrize_with_cases("fun", cases=".")
def test_invert(fun):
    key = jax.random.PRNGKey(seed=2)

    x = jax.random.normal(key, shape=(3,))
    y = fun(x)
    x_again = bijection.invert(fun)(y)
    y_again = bijection.invert(bijection.invert(fun))(x_again)

    assert jnp.allclose(x, x_again)
    assert jnp.allclose(y, y_again)


def test_invert_raises_error_for_non_invertible_function():
    def unknown_function(x):
        return x

    with pytest.raises(KeyError, match="not registered"):
        bijection.invert(unknown_function)