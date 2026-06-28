import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from config import *


# ----------------------------
# Load MNIST Dataset
# ----------------------------

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


# ----------------------------
# IID Split
# ----------------------------

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


# ----------------------------
# Test Loader
# ----------------------------

def get_test_loader(test_dataset):

    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )