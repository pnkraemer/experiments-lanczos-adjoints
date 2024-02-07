"""Test utilities."""

import os

import jax.experimental.sparse
import jax.numpy as jnp
import scipy.io
import ssgetpy


def suite_sparse_download(
    path,
    *,
    limit=5,
    isspd=None,
    nzbounds=None,
    rowbounds=(None, None),
    colbounds=(None, None),
    matrixformat="MM",
):
    """Download from https://sparse.tamu.edu/."""
    searched = ssgetpy.search(
        limit=limit,
        isspd=isspd,
        nzbounds=nzbounds,
        rowbounds=rowbounds,
        colbounds=colbounds,
    )
    searched.download(destpath=path, format=matrixformat, extract=True)


def suite_sparse_load(which, /, path="./data/matrices/"):
    matrix = scipy.io.mmread(f"{path}{which}/{which}")

    row = jnp.asarray(matrix.row)
    col = jnp.asarray(matrix.col)
    data = jnp.asarray(matrix.data)
    indices = jnp.stack([row, col]).T
    return jax.experimental.sparse.BCOO([data, indices], shape=matrix.shape)


def plt_spy_coo(ax, A, /, markersize=3, cmap="jet", invert_axes=True):
    """Plot the sparsity pattern of a BCOO matrix.

    Credit:
    https://gist.github.com/lukeolson/9710288
    """
    ax.scatter(
        A.indices[:, 0],
        A.indices[:, 1],
        c=A.data,
        s=markersize,
        marker="s",
        edgecolors="none",
        clip_on=False,
        cmap=cmap,
    )
    nrows, ncols = A.shape
    ax.set_xlim((0, nrows))
    ax.set_ylim((0, ncols))

    if invert_axes:
        ax.invert_yaxis()
        ax.xaxis.tick_top()


def matching_directory(file, where, /, replace="experiments/"):
    if where not in ["data/", "figures/"]:
        raise ValueError
    if replace not in ["experiments/"]:
        raise ValueError

    # Read directory name and replace "experiments" with e.g. "data"
    directory_file = os.path.dirname(file) + "/"
    return directory_file.replace(replace, where)
