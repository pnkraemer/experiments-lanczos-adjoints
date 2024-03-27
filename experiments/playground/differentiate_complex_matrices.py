import jax
import jax.flatten_util
from matfree_extensions import arnoldi


def main(nrows=2, krylov_depth=2, reortho="none"):
    # Create a matrix and a direction as a test-case
    A1 = jax.random.normal(jax.random.PRNGKey(1), shape=(nrows, nrows))
    A2 = jax.random.normal(jax.random.PRNGKey(2), shape=(nrows, nrows))
    v1 = jax.random.normal(jax.random.PRNGKey(3), shape=(nrows,))
    v2 = jax.random.normal(jax.random.PRNGKey(4), shape=(nrows,))
    A = jax.lax.complex(A1, A2)
    v = jax.lax.complex(v1, v2)

    reortho_fwd = {"full": "full", "none": "none", "full_with_sigma": "full"}
    reortho_ = reortho_fwd[reortho]

    def fwd(vector, params):
        return arnoldi.forward(
            lambda s, p: p @ s, krylov_depth, vector, params, reortho=reortho_
        )

    # Forward pass
    (Q, H, r, c), vjp = jax.vjp(fwd, v, A)

    # Random input gradients (no sparsity at all)
    (dQ, dH, dr, dc) = _random_like(Q, H, r, c)

    # Call the auto-diff VJP
    dv_ref, dp_ref = vjp((dQ, dH, dr, dc))

    # Call the custom VJP
    (dv, dp), _ = arnoldi.adjoint(
        lambda s, p: p @ s,
        A,
        Q=Q,
        H=H,
        r=r,
        c=c,
        dQ=dQ,
        dH=dH,
        dr=dr,
        dc=dc,
        reortho=reortho,
    )
    return (dv, dp), (dv_ref, dp_ref)


def _random_like(*tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)

    flat1 = jax.random.normal(jax.random.PRNGKey(10), shape=flat.shape)
    flat2 = jax.random.normal(jax.random.PRNGKey(11), shape=flat.shape)
    flat_like = jax.lax.complex(flat1, flat2)
    return unflatten(flat_like)


if __name__ == "__main__":
    (dv, dp), (dv_ref, dp_ref) = main()

    print(dp)
    print()
    print(dp_ref)
