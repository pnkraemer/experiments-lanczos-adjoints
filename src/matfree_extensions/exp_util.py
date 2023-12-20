"""Test utilities."""

import os

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


def plt_spy_coo(ax, A, /, markersize=3, cmap="jet"):
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
    ax.invert_yaxis()
    ax.xaxis.tick_top()


def create_matching_directory(file, where, /, replace="experiments/"):
    if where not in ["data/", "figures/"]:
        raise ValueError
    if replace not in ["experiments/"]:
        raise ValueError

    # Read directory name and replace "experiments" with e.g. "data"
    directory_file = os.path.dirname(file) + "/"
    target = directory_file.replace(replace, where)

    # Create directory unless exists
    if not os.path.exists(target):
        print(f"\nCreating {target}...\n")
        os.mkdir(target)
