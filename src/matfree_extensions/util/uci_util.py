"""UCI data loaders.

Adapted from Kaiwen Wu's code at:
https://github.com/kayween/alternating-projection-for-gp/blob/main/datasets/uci/

The adaptations are:
- Port to JAX
- Replace Dataset objects with load()-style functions.
"""

import functools
import os
from io import BytesIO
from typing import Callable
from zipfile import ZipFile

import jax.numpy as jnp
import pandas as pd
import requests  # type: ignore
import scipy.io
import ucimlrepo


def _use_cache_if_possible_otherwise_download_and_cache(name: str):
    """Create a decorator for loading vs caching a UCI dataset."""

    def wrap(load: Callable):
        """Augment a data loader with a load-vs-cache decision."""

        @functools.wraps(load)
        def load_wrapped():
            """Load the data."""

            path = f"./data/uci_processed/{name}"
            if os.path.exists(path):
                inputs = jnp.load(f"{path}/inputs.npy")
                targets = jnp.load(f"{path}/targets.npy")
                return inputs, targets

            print("\nDownloading...")
            X, y = load()
            print("done.\n")

            print("\nSaving...")
            os.makedirs(path, exist_ok=True)
            jnp.save(f"{path}/inputs.npy", X)
            jnp.save(f"{path}/targets.npy", y)
            print("done.\n")
            return X, y

        return load_wrapped

    return wrap


@_use_cache_if_possible_otherwise_download_and_cache("road_network")
def uci_road_network():
    path_get = "./data/uci_mat/3droad.mat"
    data = scipy.io.loadmat(path_get)["data"]
    inputs, targets = data[:, :-1], data[:, -1]

    # Normalise inputs
    mean = inputs.mean(axis=-2, keepdims=True)
    std = inputs.std(axis=-2, keepdims=True) + 1e-6  # prevent dividing by 0
    inputs = (inputs - mean) / std

    # Normalise targets
    mean, std = targets.mean(), targets.std()
    targets = (targets - mean) / std
    return inputs, targets


@_use_cache_if_possible_otherwise_download_and_cache("song")
def uci_song():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/"
    r = requests.get(url + "YearPredictionMSD.txt.zip")
    files = ZipFile(BytesIO(r.content))
    df = pd.read_csv(files.open("YearPredictionMSD.txt"), header=None)
    raw_data = jnp.asarray(df.values)
    X = raw_data[:, 1::]
    y = raw_data[:, 0]

    # Preprocess
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
    y = y - jnp.mean(y, axis=0)
    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("air_quality")
def uci_air_quality():
    url = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"

    r = requests.get(url)
    files = ZipFile(BytesIO(r.content))

    data_zip_file = ZipFile(BytesIO(files.read("PRSA2017_Data_20130301-20170228.zip")))

    lst_frames = []
    for name in data_zip_file.namelist():
        if name[-4:] == ".csv":
            lst_frames.append(pd.read_csv(data_zip_file.open(name)))
    assert len(lst_frames) == 12
    df = pd.concat(lst_frames)

    # drop missing data
    df.dropna(inplace=True)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df["station"] = le.fit_transform(df["station"])
    df["wd"] = le.fit_transform(df["wd"])

    # drop index column
    df.drop(["No"], axis=1, inplace=True)

    # drop PM 10, as it may be highly co-related with PM 2.5
    df.drop(["PM10"], axis=1, inplace=True)

    # drop year and day, as it may be not informative with the prediction
    df.drop(["year"], axis=1, inplace=True)
    df.drop(["day"], axis=1, inplace=True)

    X = df.drop(["PM2.5"], axis=1)
    y = df["PM2.5"]

    X = jnp.asarray(X.values)
    y = jnp.asarray(y.values)
    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("bike_sharing")
def uci_bike_sharing():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    r = requests.get(url + "Bike-Sharing-Dataset.zip")
    files = ZipFile(BytesIO(r.content))

    # Read data for the hourly count
    df = pd.read_csv(files.open("hour.csv"))

    # Convert dates to numeric
    df["dteday"] = pd.to_datetime(df["dteday"]).astype(int)

    raw_data = jnp.asarray(df.values)
    X = raw_data[:, :-1]
    y = raw_data[:, -1]
    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("kegg_undirected")
def uci_kegg_undirected():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00221/"
    df = pd.read_csv(
        url + "Reaction%20Network%20(Undirected).data", index_col=0, header=None
    )
    df.drop(df[df[4] == "?"].index, inplace=True)
    df[4] = df[4].astype(float)
    df.drop(df[df[21] > 1].index, inplace=True)
    df.drop(columns=[10], inplace=True)

    raw_data = jnp.asarray(df.values)

    # Preprocess
    X = raw_data[:, 0:-1]
    y = raw_data[:, -1]

    # Transform outputs
    y = jnp.log(y)
    y = y - jnp.mean(y, axis=0)

    # Normalize features
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("parkinson")
def uci_parkinson():
    # Read data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/"
    df = pd.read_csv(url + "parkinsons_updrs.data")
    df.drop(["motor_UPDRS"], axis=1)

    # Move column to predict
    column_to_move = df.pop("total_UPDRS")
    df.insert(0, "total_UPDRS", column_to_move)

    raw_data = jnp.asarray(df.values)

    # Preprocess
    X = raw_data[:, 1::]
    y = raw_data[:, 0]

    # Transform outputs
    y = y - jnp.mean(y, axis=0)

    # Normalize features
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("protein")
def uci_protein():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/"

    # Read data
    df = pd.read_csv(url + "CASP.csv")

    raw_data = jnp.asarray(df.values)

    # Preprocess
    X = raw_data[:, 1::]
    y = raw_data[:, 0]

    # Transform outputs
    y = jnp.log(y + 1)
    y = y - jnp.mean(y, axis=0)

    # Normalize features
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("sgemm")
def uci_sgemm():
    url = (
        "https://archive.ics.uci.edu/static/public/440/sgemm+gpu+kernel+performance.zip"
    )

    r = requests.get(url)
    files = ZipFile(BytesIO(r.content))

    df = pd.read_csv(files.open("sgemm_product.csv"))

    df["Runtime"] = df[[f"Run{i:d} (ms)" for i in (1, 2, 3, 4)]].mean(axis=1)
    df.drop(columns=[f"Run{i:d} (ms)" for i in (1, 2, 3, 4)], axis=1, inplace=True)

    raw_data = jnp.asarray(df.values)
    X = raw_data[:, :-1]
    y = raw_data[:, -1]
    y = jnp.log(y)
    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("concrete")
def uci_concrete():
    dataset = ucimlrepo.fetch_ucirepo(id=165)

    # data (as pandas dataframes)
    X = jnp.asarray(dataset.data.features.values)
    y = jnp.asarray(dataset.data.targets.values).squeeze()

    # Normalise
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
    y = y - jnp.mean(y)
    return X, y


@_use_cache_if_possible_otherwise_download_and_cache("power_plant")
def uci_power_plant():
    dataset = ucimlrepo.fetch_ucirepo(id=294)

    # data (as pandas dataframes)
    X = jnp.asarray(dataset.data.features.values)
    y = jnp.asarray(dataset.data.targets.values).squeeze()

    # Normalise
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
    y = y - jnp.mean(y)
    return X, y
