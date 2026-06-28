import os
import random
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from dataset import (
    load_datasets,
    split_clients_iid,
    get_test_loader
)

from client import Client
from server import Server
from utils import evaluate
from attacks import apply_attack

from config import *


def main():

    start_time = time.time()

    os.makedirs("results", exist_ok=True)

    # ----------------------------------
    # Load Dataset
    # ----------------------------------

    train_dataset, test_dataset = load_datasets()

    client_loaders = split_clients_iid(train_dataset)

    test_loader = get_test_loader(test_dataset)

    # ----------------------------------
    # Create Clients
    # ----------------------------------

    clients = []

    for i, loader in enumerate(client_loaders):

        clients.append(
            Client(i, loader)
        )

    # ----------------------------------
    # Create Server
    # ----------------------------------

    server = Server()

    criterion = nn.CrossEntropyLoss()

    metrics = {
        "accuracy": [],
        "loss": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    best_accuracy = 0

    # ----------------------------------
    # Random Malicious Clients
    # ----------------------------------

    malicious_clients = set(
        random.sample(
            range(NUM_CLIENTS),
            MALICIOUS_CLIENTS
        )
    )

    print("=" * 60)
    print("Malicious Clients :", malicious_clients)
    print("Attack Type       :", ATTACK_TYPE)
    print("=" * 60)

    # ----------------------------------
    # Federated Learning
    # ----------------------------------

    for round_num in range(ROUNDS):

        print("\n" + "=" * 50)

        print(f"Round {round_num + 1}")

        global_weights = server.get_weights()

        client_updates = []

        for client in clients:

            client.set_weights(global_weights)

            update = client.train()

            if client.client_id in malicious_clients:

                update = apply_attack(
                    update,
                    ATTACK_TYPE
                )

            client_updates.append(update)

        server.aggregate(client_updates)

        accuracy, loss = evaluate(
            server.get_model(),
            test_loader,
            criterion
        )

        metrics["accuracy"].append(accuracy)
        metrics["loss"].append(loss)
        metrics["precision"].append(None)
        metrics["recall"].append(None)
        metrics["f1"].append(None)

        print(f"Accuracy : {accuracy:.2f}%")

        print(f"Loss     : {loss:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                server.get_model().state_dict(),
                f"results/best_model_{ATTACK_TYPE}_{MALICIOUS_CLIENTS}.pth"
            )

    print("\n" + "=" * 60)

    print("Training Complete!")

    print(f"Best Accuracy : {best_accuracy:.2f}%")

    training_time = time.time() - start_time

    final_accuracy = metrics["accuracy"][-1] if metrics["accuracy"] else 0.0

    final_loss = metrics["loss"][-1] if metrics["loss"] else 0.0

    results = pd.DataFrame({
        "Round": range(1, ROUNDS + 1),
        "Accuracy": metrics["accuracy"],
        "Loss": metrics["loss"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"]
    })

    results.to_csv(
        f"results/results_{ATTACK_TYPE}_{MALICIOUS_CLIENTS}.csv",
        index=False
    )

    with open(
        f"results/summary_{ATTACK_TYPE}_{MALICIOUS_CLIENTS}.txt",
        "w"
    ) as f:

        f.write(f"Attack Type: {ATTACK_TYPE}\n")
        f.write(f"Malicious Clients: {MALICIOUS_CLIENTS}\n")
        f.write(f"Malicious Client IDs: {sorted(malicious_clients)}\n")
        f.write(f"Rounds: {ROUNDS}\n")
        f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Final Accuracy: {final_accuracy:.2f}%\n")
        f.write(f"Final Loss: {final_loss:.4f}\n")
        f.write(f"Training Time: {training_time:.2f} sec\n")

    # ----------------------------------
    # Accuracy Plot
    # ----------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        range(1, ROUNDS + 1),
        metrics["accuracy"],
        marker="o"
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Accuracy (%)")

    plt.title("FedAvg Accuracy")

    plt.grid(True)

    plt.savefig(
        f"results/accuracy_{ATTACK_TYPE}_{MALICIOUS_CLIENTS}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ----------------------------------
    # Loss Plot
    # ----------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        range(1, ROUNDS + 1),
        metrics["loss"],
        marker="o"
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Loss")

    plt.title("FedAvg Loss")

    plt.grid(True)

    plt.savefig(
        f"results/loss_{ATTACK_TYPE}_{MALICIOUS_CLIENTS}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n========== EXPERIMENT SUMMARY ==========")
    print(f"Attack Type       : {ATTACK_TYPE}")
    print(f"Malicious Clients : {sorted(malicious_clients)}")
    print(f"Best Accuracy     : {best_accuracy:.2f}%")
    print(f"Training Time     : {training_time:.2f} sec")
    print("========================================")


if __name__ == "__main__":

    main()