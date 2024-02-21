import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg
from matfree_extensions import arnoldi, exp_util

jnp.set_printoptions(1, suppress=False)

seed = 1
key = jax.random.PRNGKey(seed)

nrows = 5
matrix = exp_util.hilbert(nrows)
vector = jnp.ones((nrows,)) / jnp.sqrt(nrows)


reortho = "full"


def matvec(x, p):
    return p @ x


krylov_depth = len(vector)
Q, H, r, c = arnoldi.forward(matvec, krylov_depth, vector, matrix, reortho=reortho)
dQ, dH, dr, dc = exp_util.tree_random_like(key, (Q, H, r, c))

kwargs_fwd = {"Q": Q, "H": H, "r": r, "c": c}
kwargs_bwd = {"dQ": dQ, "dH": dH, "dr": dr, "dc": dc}
_, multipliers = arnoldi.adjoint(
    matvec, matrix, **kwargs_fwd, **kwargs_bwd, reortho=reortho
)
received = multipliers["Lambda"].T @ Q
expected = dH.T + jnp.triu(multipliers["Sigma"].T, 2)

print(received - expected)
