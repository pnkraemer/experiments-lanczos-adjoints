"""Test utilities."""

import os
from typing import Literal, get_args

import jax.experimental.sparse
import jax.numpy as jnp
import scipy.io


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
    import ssgetpy

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


# Rationale for choosing the data sets below (from UCI):
#
# * Multivariate data
# * Regression
# * Less than 10 features
# * More than 1000 instances
# * Numerical features
# * Available for python import
#
# This gives three sets:
#   concrete_compressive_strength,
#   combined cycle power plant,
#   and household electric
# todo: household electric dataset
UCI_DATASET_ARGS = Literal[
    "concrete_compressive_strength", "combined_cycle_power_plant"
]
UCI_DATASETS = get_args(UCI_DATASET_ARGS)


def uci_dataset(which: UCI_DATASET_ARGS, /, *, use_cache_if_possible: bool = True):
    """Load one of the UCI datasets."""

    from ucimlrepo import fetch_ucirepo

    if which not in UCI_DATASETS:
        msg = "The dataset is unknown."
        msg += f"\n\tExpected: One of {UCI_DATASETS}."
        msg += f"\n\tReceived: '{which}'."
        raise ValueError(msg)

    path = f"./data/uci_processed/{which}"
    if os.path.exists(path) and use_cache_if_possible:
        inputs = jnp.load(f"{path}/inputs.npy")
        targets = jnp.load(f"{path}/targets.npy")
        return inputs, targets

    # fetch dataset
    dataset = fetch_ucirepo(id=165)

    # data (as pandas dataframes)
    X = jnp.asarray(dataset.data.features.values)
    y = jnp.asarray(dataset.data.targets.values)

    os.makedirs(path, exist_ok=True)
    jnp.save(f"{path}/inputs.npy", X)
    jnp.save(f"{path}/targets.npy", y)
    return X, y


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
    if where not in ["data/", "figures/", "results/"]:
        raise ValueError
    if replace not in ["experiments/"]:
        raise ValueError

    # Read directory name and replace "experiments" with e.g. "data"
    directory_file = os.path.dirname(file) + "/"
    return directory_file.replace(replace, where)


def hilbert(ndim, /):
    a = jnp.arange(ndim)
    return 1 / (1 + a[:, None] + a[None, :])


def tree_random_like(key, tree, *, generate_func=jax.random.normal):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    flat_like = generate_func(key, shape=flat.shape, dtype=flat.dtype)
    return unflatten(flat_like)


def goldstein_price(X, Y, /):
    # Motivated by:
    # https://docs.jaxgaussianprocesses.com/api/decision_making/test_functions/continuous_functions/
    # See: https://www.sfu.ca/~ssurjano/goldpr.html
    # Code from: https://gist.github.com/MiguelAngelHFlores/777062e58419e1458a1c1800d00b03d5
    return (
        1 + (X + Y + 1) ** 2 * (19 - 14 * X + 3 * X**2 - 14 * Y + 6 * X * Y + 3 * Y**2)
    ) * (
        30
        + (2 * X - 3 * Y) ** 2
        * (18 - 32 * X + 12 * X**2 + 48 * Y - 36 * X * Y + 27 * Y**2)
    )
