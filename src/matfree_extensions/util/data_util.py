"""Data downloading and pre-processing utilities."""

import os
import random
from math import floor
from typing import Literal, get_args

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from scipy.io import loadmat
from torchvision import datasets
from torchvision import transforms as T

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def uci_preprocessing(inputs, targets, seed, device="cpu", verbose=False):
    """Pre-processing of UCI data as in to Wu et al. (AISTATS'24)"""

    # Pre-processing and train-test splitting (manual)
    X = torch.from_numpy(inputs)
    y = torch.from_numpy(targets)

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

    if which not in UCI_DATASETS:
        msg = "The dataset is unknown."
        msg += f"\n\tExpected: One of {UCI_DATASETS}."
        msg += f"\n\tReceived: '{which}'."
        raise ValueError(msg)

    path = f"./data/uci_processed/{which}"
    if os.path.exists(path) and use_cache_if_possible:
        inputs = np.load(f"{path}/inputs.npy")
        targets = np.load(f"{path}/targets.npy")
        inputs_train, targets_train, inputs_test, targets_test = uci_preprocessing(
            inputs, targets, seed, device, verbose
        )
        return (inputs_train, targets_train), (inputs_test, targets_test)

    if (
        which == "airline"
        or which == "airquality"
        or (which == "sgemmgpu" or which == "gassensors")
    ):
        # to-do as in: https://github.com/kayween/alternating-projection-for-gp/blob/main/data.py
        pass
    elif (
        which == "concrete_compressive_strength"
        or which == "combined_cycle_power_plant"
    ):
        from ucimlrepo import fetch_ucirepo

        # fetch dataset
        dataset = fetch_ucirepo(id=165)  # same id for both?

        # data (as pandas dataframes)
        X_full = np.asarray(dataset.data.features.values)
        y_full = np.asarray(dataset.data.targets.values)

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


def numpy_collate_fn(batch):
    data, target = zip(*batch)
    data = np.stack(data)
    target = np.stack(target)
    return {"image": data, "label": target}


def get_cifar10(
    batch_size=128, seed=0, download: bool = True, data_path="/dtu/p1/hroy/data"
):
    n_classes = 10
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=download
    )
    means = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    std = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    test_transform = T.Compose([T.ToTensor(), T.Normalize(means, std)])
    # For training, we add some augmentation.
    # Networks are too powerful and would overfit.
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(means, std),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, transform=train_transform, download=download
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, transform=test_transform, download=download
    )
    set_seed(seed)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    set_seed(seed)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = torchvision.datasets.CIFAR10(
        root=data_path, train=False, transform=test_transform, download=download
    )
    train_set.dataset.targets = torch.nn.functional.one_hot(
        torch.tensor(train_set.dataset.targets), n_classes
    ).numpy()
    val_set.dataset.targets = torch.nn.functional.one_hot(
        torch.tensor(val_set.dataset.targets), n_classes
    ).numpy()
    test_set.targets = torch.nn.functional.one_hot(
        torch.tensor(test_set.targets), n_classes
    ).numpy()
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )

    return train_loader, val_loader, test_loader


def get_cifar100(
    batch_size=128, seed=0, download: bool = True, data_path="/dtu/p1/hroy/data"
):
    n_classes = 100
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_path, train=True, download=download
    )
    means = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    std = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    test_transform = T.Compose([T.ToTensor(), T.Normalize(means, std)])
    # For training, we add some augmentation.
    # Networks are too powerful and would overfit.
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(means, std),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_path, train=True, transform=train_transform, download=download
    )
    val_dataset = torchvision.datasets.CIFAR100(
        root=data_path, train=True, transform=test_transform, download=download
    )
    set_seed(seed)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    set_seed(seed)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = torchvision.datasets.CIFAR100(
        root=data_path, train=False, transform=test_transform, download=download
    )
    train_set.dataset.targets = torch.nn.functional.one_hot(
        torch.tensor(train_set.dataset.targets), n_classes
    ).numpy()
    val_set.dataset.targets = torch.nn.functional.one_hot(
        torch.tensor(val_set.dataset.targets), n_classes
    ).numpy()
    test_set.targets = torch.nn.functional.one_hot(
        torch.tensor(test_set.targets), n_classes
    ).numpy()
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )

    return train_loader, val_loader, test_loader


def image_to_numpy(mean, std):
    def normalize(img):
        img = np.array(img, dtype=np.float32)
        # img = (img / 255.0 - mean) / std
        return (img / 255.0 - mean) / std

    return normalize


def ImageNet1k_loaders(batch_size: int = 128, seed: int = 0):
    set_seed(seed)
    n_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = image_to_numpy(mean, std)
    train_transform = T.Compose(
        [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), normalize]
    )

    # test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), normalize])
    def target_transform(y):
        return F.one_hot(torch.tensor(y), n_classes).numpy()

    # target_transform = lambda y: F.one_hot(torch.tensor(y), n_classes).numpy()
    train_path = "/dtu/imagenet/ILSVRC/Data/CLS-LOC/train/"
    train_dataset = datasets.ImageFolder(
        train_path, transform=train_transform, target_transform=target_transform
    )
    return data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=numpy_collate_fn,
    )
    # return train_loader
