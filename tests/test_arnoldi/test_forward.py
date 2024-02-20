import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
import pytest_cases
from matfree_extensions import arnoldi, exp_util


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize("krylov_depth", [1, 5, 10])
@pytest_cases.parametrize("reortho", ["none", "full"])
def test_decomposition(nrows, krylov_depth, reortho):
    # Create a well-conditioned test-matrix
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(nrows, nrows))
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    # Decompose
    Q, H, r, c = arnoldi.forward(
        lambda s, p: p @ s, v, krylov_depth, A, reortho=reortho
    )

    # Assert shapes
    assert Q.shape == (nrows, krylov_depth)
    assert H.shape == (krylov_depth, krylov_depth)
    assert r.shape == (nrows,)
    assert c.shape == ()

    # Tie the test-strictness to the floating point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Test the decompositions
    e0, ek = jnp.eye(krylov_depth)[[0, -1], :]
    assert jnp.allclose(A @ Q - Q @ H - jnp.outer(r, ek), 0.0, **tols)
    assert jnp.allclose(Q.T @ Q - jnp.eye(krylov_depth), 0.0, **tols)
    assert jnp.allclose(Q @ e0, c * v, **tols)


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize("krylov_depth", [1, 5, 10])
@pytest_cases.parametrize("reortho", ["full"])
def test_reorthogonalisation(nrows, krylov_depth, reortho):
    # Create an ill-conditioned test-matrix (that requires reortho=True)
    A = exp_util.hilbert(nrows)
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,))

    # Decompose
    Q, H, r, c = arnoldi.forward(
        lambda s, p: p @ s, v, krylov_depth, A, reortho=reortho
    )

    # Assert shapes
    assert Q.shape == (nrows, krylov_depth)
    assert H.shape == (krylov_depth, krylov_depth)
    assert r.shape == (nrows,)
    assert c.shape == ()

    # Tie the test-strictness to the floating point accuracy
    small_value = jnp.sqrt(jnp.finfo(jnp.dtype(H)).eps)
    tols = {"atol": small_value, "rtol": small_value}

    # Test the decompositions
    e0, ek = jnp.eye(krylov_depth)[[0, -1], :]
    assert jnp.allclose(A @ Q - Q @ H - jnp.outer(r, ek), 0.0, **tols)
    assert jnp.allclose(Q.T @ Q - jnp.eye(krylov_depth), 0.0, **tols)
    assert jnp.allclose(Q @ e0, c * v, **tols)


def test_forward_raises_error_for_wrong_depth_too_small():
    with pytest.raises(ValueError, match="depth"):
        _ = arnoldi.forward(lambda v: v, jnp.ones((2,)), 0, reortho="none")


def test_forward_raises_value_error_for_wrong_depth_too_high():
    with pytest.raises(ValueError, match="depth"):
        _ = arnoldi.forward(lambda v: v, jnp.ones((2,)), 3, reortho="none")


@pytest_cases.parametrize("reortho", [True, "full_with_sparsity", "None"])
def test_forward_raises_type_error_for_wrong_reorthogonalisation_flag(reortho):
    one = jnp.ones((1,))
    with pytest.raises(TypeError, match="Unexpected input"):
        arnoldi.forward(lambda s: s, one, 1, reortho=reortho)
