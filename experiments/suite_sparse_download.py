# import ssgetpy
# #
# import scipy.
# matrix = ssgetpy.search(isspd=True,dtype="real")#.download(extract=True, format="MAT")
#
# print(matrix)

import jax.experimental.sparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.io

matrix = scipy.io.mmread("matrices/MM/HB/1138_bus/1138_bus.mtx")

# print(matrix)
indices = jnp.stack([matrix.row, matrix.col]).T

matrix = jax.experimental.sparse.BCOO((matrix.data, indices), shape=matrix.shape)
M = matrix.todense()

x = jnp.ones((matrix.shape[1],))

fun1 = jax.jit(
    lambda a, b: jax.experimental.sparse.BCOO([b, indices], shape=matrix.shape) @ a
)
fun2 = jax.jit(lambda a, b: b @ a)

data = jnp.asarray(matrix.data)
# %timeit fun1(x, data).block_until_ready()
# %timeit fun2(x, M).block_until_ready()

# next up: using 1138_bus, construct logdet estimators
plt.spy(matrix.todense())
plt.show()
print(matrix)
