import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest_cases
from matfree_extensions import arnoldi


@pytest_cases.parametrize("nrows", [10])
@pytest_cases.parametrize("krylov_depth", [1, 5, 10])
@pytest_cases.parametrize("reortho", ["none", "full"])
@pytest_cases.parametrize("dtype", [float, complex])
def test_decomposition_is_satisfied(nrows, krylov_depth, reortho, dtype):
    # Create a well-conditioned test-matrix
    A = jax.random.normal(jax.random.PRNGKey(1), shape=(nrows, nrows), dtype=dtype)
    v = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows,), dtype=dtype)

    # Decompose
    algorithm = arnoldi.hessenberg(lambda s, p: p @ s, krylov_depth, reortho=reortho)
    Q, H, r, c = algorithm(v, A)

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
    assert jnp.allclose(Q.T.conj() @ Q - jnp.eye(krylov_depth), 0.0, **tols)
    assert jnp.allclose(Q @ e0, c * v, **tols)
