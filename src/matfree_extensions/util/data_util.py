"""Data downloading and pre-processing utilities."""

import os
import random
from io import BytesIO
from math import floor
from typing import Literal, get_args
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch
from scipy.io import loadmat

UCI_DATASET_ARGS = Literal[
    "concrete_compressive_strength",
    "combined_cycle_power_plant",
    "airline",
    "airquality",
    "sgemmgpu",
    "gassensors",
]
UCI_DATASETS = get_args(UCI_DATASET_ARGS)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def uci_preprocessing(inputs, targets, seed, device="cpu", verbose=False):
    """Pre-processing of UCI data as in to Wu et al. (AISTATS'24)"""

    # Pre-processing and train-test splitting (manual)
    X = torch.from_numpy(inputs)
    y = torch.from_numpy(targets).squeeze()

    set_seed(seed)

    good_dimensions = X.var(dim=-2) > 1.0e-10
    if int(good_dimensions.sum()) < X.size(1):
        if verbose:
            print(
                "Removed %d dimensions with no variance"
                % (X.size(1) - int(good_dimensions.sum()))
            )
        X = X[:, good_dimensions]

    X = X - X.min(0)[0]
    X = 2.0 * (X / X.max(0)[0]) - 1.0
    y -= y.mean()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    N_train = int(floor(0.8 * X.size(0)))
    N_valid = 0

    # X_train = X[:N_train, :].contiguous().to(device)
    # y_train = y[:N_train].contiguous().to(device)

    X_train = X[:N_train, :].contiguous().to(device)
    y_train = y[:N_train].contiguous().to(device)

    # valid_x = X[train_n: train_n + valid_n, :].contiguous().to(device)
    # valid_y = y[train_n: train_n + valid_n].contiguous().to(device)

    X_test = X[N_train + N_valid :, :].contiguous().to(device)
    y_test = y[N_train + N_valid :].contiguous().to(device)

    if verbose:
        print(f"Loaded data with input dimension of {X_test.size(-1)}")

    return X_train, y_train, X_test, y_test


def load_uci_data(
    which: UCI_DATASET_ARGS,
    seed: int = 1,
    num_points: int = -1,
    use_cache_if_possible: bool = True,
    device="cpu",
    verbose=False,
):
    """Loads and pre-processed the UCI data"""

    device = torch.device("cuda:0")

    if which not in UCI_DATASETS:
        msg = "The dataset is unknown."
        msg += f"\n\tExpected: One of {UCI_DATASETS}."
        msg += f"\n\tReceived: '{which}'."
        raise ValueError(msg)

    path = f"./data/uci_processed/{which}"
    if os.path.exists(path) and use_cache_if_possible:
        inputs = np.load(f"{path}/inputs.npy", allow_pickle=True)
        targets = np.load(f"{path}/targets.npy", allow_pickle=True)
        inputs_train, targets_train, inputs_test, targets_test = uci_preprocessing(
            inputs, targets, seed, device, verbose
        )
        return (inputs_train, targets_train), (inputs_test, targets_test)

    if which == "airquality":
        URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
        r = requests.get(URL)
        files = ZipFile(BytesIO(r.content))

        data_zip_file = ZipFile(
            BytesIO(files.read("PRSA2017_Data_20130301-20170228.zip"))
        )

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

        # data (as pandas dataframes)
        X_full = np.asarray(X.values)
        y_full = np.asarray(y.values)

        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/inputs.npy", X_full)
        np.save(f"{path}/targets.npy", y_full)

        # torch tensors + num. of samples
        X, y = X_full[:num_points], y_full[:num_points]

    elif which == "airline" or which == "sgemmgpu" or which == "gassensors":
        # to-do as in: https://github.com/kayween/alternating-projection-for-gp/blob/main/data.py
        pass
    elif (
        which == "concrete_compressive_strength"
        or which == "combined_cycle_power_plant"
    ):
        from ucimlrepo import fetch_ucirepo
        # print(list_available_datasets())

        # fetch dataset
        # combined_cycle_power_plant -> 294
        dataset = fetch_ucirepo(id=360)  # same id for both?

        # data (as pandas dataframes)
        X_full = np.asarray(dataset.data.features.values)
        y_full = np.asarray(dataset.data.targets)

        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/inputs.npy", X_full)
        np.save(f"{path}/targets.npy", y_full)

        # torch tensors + num. of samples
        X, y = X_full[:num_points], y_full[:num_points]

    else:
        # others from .mat format?
        data = torch.Tensor(loadmat(path + f"{which}.mat")["data"])
        X = data[:num_points]
        y = data[:num_points]

    X_train, y_train, X_test, y_test = uci_preprocessing(X, y, seed, device, verbose)
    return (X_train, y_train), (X_test, y_test)
