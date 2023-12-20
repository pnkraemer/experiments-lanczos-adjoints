"""Test utilities."""

import jax.experimental.sparse
import jax.numpy as jnp
import scipy.io
import ssgetpy


def suite_sparse_download(path, format="MM"):
    ssgetpy.search().download(destpath=path, format=format, extract=True)


def suite_sparse_load(which, /, path):
    matrix = scipy.io.mmread(f"{path}{which}/{which}")

    row = jnp.asarray(matrix.row)
    col = jnp.asarray(matrix.col)
    data = jnp.asarray(matrix.data)
    indices = jnp.stack([row, col]).T
    return jax.experimental.sparse.BCOO([data, indices], shape=matrix.shape)
