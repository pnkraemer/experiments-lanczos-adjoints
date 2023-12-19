import jax
import jax.experimental.sparse
import jax.numpy as jnp
import matplotlib.pyplot as plt

from matfree_extensions import bcoo_random_spd

n = 100  # nrows/ncols
nz = 200  # nonzeros
seed = 1
key = jax.random.PRNGKey(seed)

M = bcoo_random_spd(key, num_rows=n, num_nonzeros=nz)

print(M)
print(M.todense())

x = jnp.ones((n,))
print(M @ x)
print(jnp.linalg.slogdet(M.todense()))


plt.spy(M.todense())
plt.show()
