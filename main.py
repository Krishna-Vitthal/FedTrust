import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import (
    load_datasets,
    split_clients_iid,
    get_test_loader
)

from client import Client
from server import Server
from utils import evaluate

from config import *


def main():

    # -----------------------
    # Load Dataset
    # -----------------------

    train_dataset, test_dataset = load_datasets()

    client_loaders = split_clients_iid(train_dataset)

    test_loader = get_test_loader(test_dataset)

    # -----------------------
    # Create Clients
    # -----------------------

    clients = []

    for i, loader in enumerate(client_loaders):

        clients.append(
            Client(i, loader)
        )

    # -----------------------
    # Create Server
    # -----------------------

    server = Server()

    criterion = nn.CrossEntropyLoss()

    accuracy_history = []

    loss_history = []

    best_accuracy = 0

    # -----------------------
    # Federated Learning
    # -----------------------

    for round_num in range(ROUNDS):

        print("=" * 50)

        print(f"Round {round_num + 1}")

        global_weights = server.get_weights()

        client_updates = []

        for client in clients:

            client.set_weights(global_weights)

            update = client.train()

            client_updates.append(update)

        server.aggregate(client_updates)

        accuracy, loss = evaluate(
            server.global_model,
            test_loader,
            criterion
        )

        accuracy_history.append(accuracy)

        loss_history.append(loss)

        print(f"Accuracy : {accuracy:.2f}%")

        print(f"Loss     : {loss:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                server.global_model.state_dict(),
                "best_model.pth"
            )

    print("=" * 50)

    print("Training Complete!")

    print(f"Best Accuracy : {best_accuracy:.2f}%")

    # -----------------------
    # Plot Accuracy
    # -----------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        range(1, ROUNDS + 1),
        accuracy_history,
        marker="o"
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Accuracy (%)")

    plt.title("FedAvg Accuracy")

    plt.grid(True)

    plt.savefig("accuracy.png")

    plt.close()

    # -----------------------
    # Plot Loss
    # -----------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        range(1, ROUNDS + 1),
        loss_history,
        marker="o"
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Loss")

    plt.title("FedAvg Loss")

    plt.grid(True)

    plt.savefig("loss.png")

    plt.close()


if __name__ == "__main__":

    main()