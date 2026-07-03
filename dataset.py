import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from config import *


# =====================================================
# Load MNIST Dataset
# =====================================================

def load_datasets():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


# =====================================================
# IID Split
# =====================================================

def split_clients_iid(train_dataset):

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(train_dataset),
        generator=generator
    )

    client_size = len(train_dataset) // NUM_CLIENTS

    client_loaders = []

    for i in range(NUM_CLIENTS):

        start = i * client_size

        if i == NUM_CLIENTS - 1:
            end = len(train_dataset)
        else:
            end = (i + 1) * client_size

        subset = Subset(
            train_dataset,
            indices[start:end]
        )

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        client_loaders.append(loader)

    return client_loaders


# =====================================================
# Sorted Non-IID (Extreme)
# =====================================================

def split_clients_noniid(train_dataset):

    labels = torch.tensor(train_dataset.targets)

    sorted_indices = torch.argsort(labels)

    client_size = len(train_dataset) // NUM_CLIENTS

    client_loaders = []

    for i in range(NUM_CLIENTS):

        start = i * client_size

        if i == NUM_CLIENTS - 1:
            end = len(train_dataset)
        else:
            end = (i + 1) * client_size

        subset = Subset(
            train_dataset,
            sorted_indices[start:end]
        )

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        client_loaders.append(loader)

    return client_loaders


# =====================================================
# Dirichlet Non-IID (Realistic)
# =====================================================

def split_clients_dirichlet(train_dataset):

    np.random.seed(SEED)

    labels = np.array(train_dataset.targets)

    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(NUM_CLIENTS)]

    # Allocate each class independently
    for cls in range(num_classes):

        class_indices = np.where(labels == cls)[0]

        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(

            np.repeat(
                DIRICHLET_ALPHA,
                NUM_CLIENTS
            )

        )

        split_points = (

            np.cumsum(proportions)

            * len(class_indices)

        ).astype(int)[:-1]

        class_split = np.split(
            class_indices,
            split_points
        )

        for client_id in range(NUM_CLIENTS):

            client_indices[client_id].extend(

                class_split[client_id].tolist()

            )

    client_loaders = []

    for client_id in range(NUM_CLIENTS):

        np.random.shuffle(

            client_indices[client_id]

        )

        subset = Subset(

            train_dataset,

            client_indices[client_id]

        )

        loader = DataLoader(

            subset,

            batch_size=BATCH_SIZE,

            shuffle=True

        )

        client_loaders.append(loader)

    return client_loaders


# =====================================================
# Test Loader
# =====================================================

def get_test_loader(test_dataset):

    return DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False

    )