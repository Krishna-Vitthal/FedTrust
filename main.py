import os
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from attacks import apply_attack
from client import Client
from config import *
from dataset import (
    get_test_loader,
    load_datasets,
    split_clients_noniid,
    split_clients_iid,
)
from server import Server
from utils import evaluate


def run_experiment(attack_type, malicious_clients):

    start_time = time.time()

    train_dataset, test_dataset = load_datasets()

    if DATA_SPLIT == "noniid":

        client_loaders = split_clients_noniid(train_dataset)

    else:

        client_loaders = split_clients_iid(train_dataset)

    test_loader = get_test_loader(test_dataset)

    clients = []

    for i, loader in enumerate(client_loaders):

        clients.append(Client(i, loader))

    server = Server()

    criterion = nn.CrossEntropyLoss()

    metrics = {
        "accuracy": [],
        "loss": []
    }

    best_accuracy = 0.0

    print("=" * 60)
    print(f"Attack Type       : {attack_type}")
    print(f"Malicious Clients : {sorted(malicious_clients)}")
    print("=" * 60)

    for round_num in range(ROUNDS):

        print("\n" + "=" * 50)
        print(f"Round {round_num + 1}")

        global_weights = server.get_weights()

        client_updates = []

        for client in clients:

            client.set_weights(global_weights)

            update = client.train()

            if attack_type != "none" and client.client_id in malicious_clients:

                update = apply_attack(update, attack_type)

            client_updates.append(update)

        server.aggregate(client_updates)

        accuracy, loss = evaluate(
            server.get_model(),
            test_loader,
            criterion,
        )

        metrics["accuracy"].append(accuracy)
        metrics["loss"].append(loss)
        print(f"Accuracy : {accuracy:.2f}%")
        print(f"Loss     : {loss:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                server.get_model().state_dict(),
                f"results/best_model_{attack_type}_{MALICIOUS_CLIENTS}.pth"
            )

    training_time = time.time() - start_time

    final_accuracy = metrics["accuracy"][-1] if metrics["accuracy"] else 0.0
    final_loss = metrics["loss"][-1] if metrics["loss"] else 0.0

    results = pd.DataFrame({
        "Round": range(1, ROUNDS + 1),
        "Accuracy": metrics["accuracy"],
        "Loss": metrics["loss"]
    })

    results.to_csv(
        f"results/results_{attack_type}_{MALICIOUS_CLIENTS}.csv",
        index=False
    )

    with open(
        f"results/summary_{attack_type}_{MALICIOUS_CLIENTS}.txt",
        "w",
    ) as f:

        f.write(f"Attack Type: {attack_type}\n")
        f.write(f"Malicious Clients: {MALICIOUS_CLIENTS}\n")
        f.write(f"Malicious Client IDs: {sorted(malicious_clients)}\n")
        f.write(f"Rounds: {ROUNDS}\n")
        f.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Final Accuracy: {final_accuracy:.2f}%\n")
        f.write(f"Final Loss: {final_loss:.4f}\n")
        f.write(f"Training Time: {training_time:.2f} sec\n")

    trust_history = server.get_trust_history()

    if USE_TRUST and trust_history:

        trust_df = pd.DataFrame(trust_history)

        trust_df.insert(0, "Round", range(1, len(trust_df) + 1))

        trust_df.to_csv(
            f"results/trust_history_{attack_type}_{MALICIOUS_CLIENTS}.csv",
            index=False
        )

        plt.figure(figsize=(10, 6))

        for client_id in sorted(
            column for column in trust_df.columns if column != "Round"
        ):

            values = trust_df[client_id]

            is_malicious = client_id in malicious_clients

            plt.plot(
                trust_df["Round"],
                values,
                label=f"Client {client_id}",
                linewidth=2.2 if is_malicious else 1.2,
                alpha=0.9 if is_malicious else 0.55,
                linestyle="-" if is_malicious else "--",
            )

        plt.xlabel("Communication Round")
        plt.ylabel("Trust")
        plt.ylim(0.0, 1.05)
        plt.title(f"Trust Trajectories - {attack_type}")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(
            f"results/trust_trajectory_{attack_type}_{MALICIOUS_CLIENTS}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, ROUNDS + 1), metrics["accuracy"], marker="o")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title(f"FedAvg Accuracy - {attack_type}")
    plt.grid(True)
    plt.savefig(
        f"results/accuracy_{attack_type}_{MALICIOUS_CLIENTS}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, ROUNDS + 1), metrics["loss"], marker="o")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title(f"FedAvg Loss - {attack_type}")
    plt.grid(True)
    plt.savefig(
        f"results/loss_{attack_type}_{MALICIOUS_CLIENTS}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\n========== EXPERIMENT SUMMARY ==========")
    print(f"Attack Type       : {attack_type}")
    print(f"Malicious Clients : {sorted(malicious_clients)}")
    print(f"Best Accuracy     : {best_accuracy:.2f}%")
    print(f"Training Time     : {training_time:.2f} sec")
    print("========================================")

    return {
        "attack_type": attack_type,
        "data_split": DATA_SPLIT,
        "malicious_clients": sorted(malicious_clients),
        "final_accuracy": final_accuracy,
        "best_accuracy": best_accuracy,
        "final_loss": final_loss,
        "training_time": training_time,
    }


def main():

    os.makedirs("results", exist_ok=True)

    rng = random.Random(SEED)
    num_malicious = min(MALICIOUS_CLIENTS, NUM_CLIENTS)

    malicious_clients = set(
        rng.sample(
            range(NUM_CLIENTS),
            num_malicious
        )
    )

    attack_types = [
        "none",
        "sign",
        "scaling",
        "gaussian",
        "random",
        "zero",
        "mixed",
    ]

    summaries = []

    for attack_type in attack_types:

        summaries.append(
            run_experiment(attack_type, malicious_clients)
        )

    print("\n========== ALL ATTACKS COMPLETE ==========")

    for summary in summaries:

        print(
            f"{summary['attack_type']:>8} : best={summary['best_accuracy']:.2f}% "
            f"time={summary['training_time']:.2f} sec"
        )

    summary_rows = pd.DataFrame([
        {
            "Data Split": summary["data_split"],
            "Attack": summary["attack_type"],
            "Malicious Clients": MALICIOUS_CLIENTS,
            "Malicious Client IDs": str(summary["malicious_clients"]),
            "Best Accuracy": summary["best_accuracy"],
            "Final Accuracy": summary["final_accuracy"],
            "Final Loss": summary["final_loss"],
            "Training Time": summary["training_time"],
        }
        for summary in summaries
    ])

    summary_rows.to_csv(
        "results/experiment_summary.csv",
        index=False
    )

    print("========================================")


if __name__ == "__main__":

    main()