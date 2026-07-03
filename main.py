import os
import random
import time
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import torch
import torch.nn as nn

import attacks as attacks_module
import dataset as dataset_module
import server as server_module
import trust as trust_module
import aggregation as aggregation_module

from attacks import apply_attack
from client import Client
from config import *
from dataset import (
    get_test_loader,
    load_datasets,
    split_clients_iid,
    split_clients_noniid,
    split_clients_dirichlet,
)
from server import Server
from utils import evaluate


def set_experiment_params(
    *,
    data_split,
    dirichlet_alpha,
    attack_scale,
    use_trust,
    use_trimmed_mean,
    use_trust_filter,
    similarity_weight,
    consistency_weight,
    alignment_weight,
    norm_weight,
):

    globals()["DATA_SPLIT"] = data_split
    globals()["DIRICHLET_ALPHA"] = dirichlet_alpha
    globals()["ATTACK_SCALE"] = attack_scale
    globals()["USE_TRUST"] = use_trust
    globals()["USE_TRIMMED_MEAN"] = use_trimmed_mean
    globals()["USE_TRUST_FILTER"] = use_trust_filter
    globals()["SIMILARITY_WEIGHT"] = similarity_weight
    globals()["CONSISTENCY_WEIGHT"] = consistency_weight
    globals()["ALIGNMENT_WEIGHT"] = alignment_weight
    globals()["NORM_WEIGHT"] = norm_weight

    dataset_module.DIRICHLET_ALPHA = dirichlet_alpha
    attacks_module.ATTACK_SCALE = attack_scale
    server_module.USE_TRUST = use_trust
    aggregation_module.USE_TRUST_FILTER = use_trust_filter
    trust_module.SIMILARITY_WEIGHT = similarity_weight
    trust_module.CONSISTENCY_WEIGHT = consistency_weight
    trust_module.ALIGNMENT_WEIGHT = alignment_weight
    trust_module.NORM_WEIGHT = norm_weight


def build_client_partition(train_dataset):

    if DATA_SPLIT == "iid":

        return split_clients_iid(train_dataset)

    if DATA_SPLIT == "noniid":

        return split_clients_noniid(train_dataset)

    if DATA_SPLIT == "dirichlet":

        return split_clients_dirichlet(train_dataset)

    raise ValueError(
        f"Unknown DATA_SPLIT: {DATA_SPLIT}"
    )


def select_malicious_clients(num_clients, malicious_ratio, seed_offset=0):

    rng = random.Random(SEED + seed_offset)
    num_malicious = max(1, min(num_clients, round(num_clients * malicious_ratio)))

    return set(
        rng.sample(
            range(num_clients),
            num_malicious
        )
    )


def ensure_results_dir():

    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_dataframe_artifacts(dataframe, base_name, write_latex=True):

    ensure_results_dir()

    csv_path = os.path.join(RESULTS_DIR, f"{base_name}.csv")
    dataframe.to_csv(csv_path, index=False)

    if write_latex:

        latex_path = os.path.join(RESULTS_DIR, f"{base_name}.tex")

        with open(latex_path, "w", encoding="utf-8") as file_handle:

            file_handle.write(
                dataframe.to_latex(
                    index=False,
                    float_format=lambda value: f"{value:.2f}",
                    escape=False,
                )
            )


def save_line_plot(dataframe, x_column, y_columns, labels, title, y_label, filename):

    ensure_results_dir()

    plt.figure(figsize=(8.8, 5.2))

    for y_column, label in zip(y_columns, labels):

        plt.plot(
            dataframe[x_column],
            dataframe[y_column],
            marker="o",
            linewidth=2.2,
            label=label,
        )

    plt.xlabel(x_column)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def save_grouped_bar_chart(dataframe, x_column, bar_columns, labels, title, y_label, filename):

    ensure_results_dir()

    plt.figure(figsize=(10.2, 5.6))

    x_positions = list(range(len(dataframe)))
    width = 0.8 / max(len(bar_columns), 1)

    for index, (bar_column, label) in enumerate(zip(bar_columns, labels)):

        offsets = [position + index * width for position in x_positions]
        plt.bar(offsets, dataframe[bar_column], width=width, label=label)

    center_positions = [position + width * (len(bar_columns) - 1) / 2 for position in x_positions]
    plt.xticks(center_positions, dataframe[x_column], rotation=25, ha="right")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def save_bar_chart(labels, values, title, y_label, filename, rotation=0):

    ensure_results_dir()

    plt.figure(figsize=(9.5, 5.4))
    plt.bar(labels, values, color="#2b6cb0")
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def save_stacked_dirichlet_chart(client_histograms, title, filename):

    ensure_results_dir()

    plt.figure(figsize=(11, 6))

    client_ids = sorted(client_histograms.keys())
    class_count = len(next(iter(client_histograms.values()))) if client_histograms else 0
    bottom = [0.0] * len(client_ids)
    colors = plt.cm.tab20.colors

    for class_index in range(class_count):

        values = [client_histograms[client_id][class_index] for client_id in client_ids]
        plt.bar(
            [str(client_id) for client_id in client_ids],
            values,
            bottom=bottom,
            color=colors[class_index % len(colors)],
            label=f"Class {class_index}",
        )
        bottom = [existing + value for existing, value in zip(bottom, values)]

    plt.title(title)
    plt.ylabel("Class Proportion")
    plt.xlabel("Client")
    plt.ylim(0.0, 1.05)
    plt.legend(ncol=2, frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def save_flow_diagram(steps, title, filename):

    ensure_results_dir()

    plt.figure(figsize=(12, 3.4))
    axis = plt.gca()
    axis.axis("off")

    step_count = len(steps)
    x_positions = [0.08 + i * (0.84 / max(step_count - 1, 1)) for i in range(step_count)]
    for index, (x_position, step) in enumerate(zip(x_positions, steps)):
        box = mpatches.FancyBboxPatch(
            (x_position - 0.055, 0.38),
            0.11,
            0.24,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="#1f2937",
            facecolor="#e2e8f0",
        )
        axis.add_patch(box)
        axis.text(x_position, 0.50, step, ha="center", va="center", fontsize=10)
        if index < step_count - 1:
            axis.annotate(
                "",
                xy=(x_position + 0.06, 0.50),
                xytext=(x_position + 0.13, 0.50),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#4a5568"),
            )

    axis.text(0.5, 0.88, title, ha="center", va="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def rows_from_score_history(summary, key_name):

    rows = []

    for round_index, score_map in enumerate(summary.get(key_name, []), start=1):

        for client_id, score in score_map.items():

            rows.append(
                {
                    "Experiment": summary["experiment_name"],
                    "Attack": summary["attack_type"],
                    "Data Split": summary["data_split"],
                    "Round": round_index,
                    "Client": client_id,
                    "Score": score,
                }
            )

    return rows


def run_experiment(
    attack_type,
    malicious_clients,
    *,
    experiment_name=None,
    collect_diagnostics=False,
):

    start_time = time.time()

    train_dataset, test_dataset = load_datasets()

    # =====================================================
    # Dataset Split Selection
    # =====================================================

    client_loaders = build_client_partition(train_dataset)

    test_loader = get_test_loader(test_dataset)

    clients = []

    for i, loader in enumerate(client_loaders):

        clients.append(Client(i, loader))

    # =====================================================
    # Display Client Data Distribution
    # =====================================================

    print("\n" + "=" * 70)
    print(f"Dataset Split : {DATA_SPLIT}")
    print("=" * 70)

    for client in clients:

        print(
            f"Client {client.client_id} "
            f"Histogram : "
            f"{[round(x, 3) for x in client.label_histogram]}"
        )

    print("=" * 70 + "\n")

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
    if experiment_name:
        print(f"Experiment        : {experiment_name}")
    print("=" * 60)

    run_tag = experiment_name or f"{attack_type}_{len(malicious_clients)}"

    round_diagnostics = []
    accuracy_history = []
    loss_history = []
    trust_history_rows = []
    behavior_history_rows = []
    similarity_history_rows = []
    consistency_history_rows = []
    alignment_history_rows = []
    norm_history_rows = []
    filtered_history_rows = []

    current_trust_scores = {}
    current_behavior_scores = {}
    current_similarity_scores = {}
    current_consistency_scores = {}
    current_alignment_scores = {}
    current_norm_scores = {}

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

        aggregation_info = server.aggregate(client_updates)

        if aggregation_info:
            round_diagnostics.append({
                "Round": round_num + 1,
                "Attack": attack_type,
                "Filtered Count": aggregation_info.get("filtered_out_count", 0),
                "Total Clients": aggregation_info.get("total_clients", len(client_updates)),
                "Filtered IDs": str(aggregation_info.get("filtered_out_client_ids", [])),
            })

            filtered_history_rows.append(
                {
                    "Experiment": run_tag,
                    "Attack": attack_type,
                    "Round": round_num + 1,
                    "Filtered Clients": aggregation_info.get("filtered_out_count", 0),
                    "Kept Clients": aggregation_info.get("filtered_count", len(client_updates)),
                    "Total Clients": aggregation_info.get("total_clients", len(client_updates)),
                    "Filtered IDs": str(aggregation_info.get("filtered_out_client_ids", [])),
                }
            )

        accuracy, loss = evaluate(
            server.get_model(),
            test_loader,
            criterion,
        )

        metrics["accuracy"].append(accuracy)
        metrics["loss"].append(loss)
        accuracy_history.append(
            {
                "Experiment": run_tag,
                "Attack": attack_type,
                "Round": round_num + 1,
                "Accuracy": accuracy,
            }
        )
        loss_history.append(
            {
                "Experiment": run_tag,
                "Attack": attack_type,
                "Round": round_num + 1,
                "Loss": loss,
            }
        )
        print(f"Accuracy : {accuracy:.2f}%")
        print(f"Loss     : {loss:.4f}")

        current_trust_scores = server.get_trust_scores()
        current_behavior_scores = server.get_behavior_scores()
        current_similarity_scores = server.get_similarity_scores()
        current_consistency_scores = server.get_consistency_scores()
        current_alignment_scores = server.get_alignment_scores()
        current_norm_scores = server.get_norm_scores()

        if USE_TRUST and current_trust_scores:

            for client_id, score in current_trust_scores.items():

                trust_history_rows.append(
                    {
                        "Experiment": run_tag,
                        "Attack": attack_type,
                        "Round": round_num + 1,
                        "Client": client_id,
                        "Trust": score,
                    }
                )

        if current_behavior_scores:

            for client_id, score in current_behavior_scores.items():

                behavior_history_rows.append(
                    {
                        "Experiment": run_tag,
                        "Attack": attack_type,
                        "Round": round_num + 1,
                        "Client": client_id,
                        "Behavior": score,
                    }
                )

        if current_similarity_scores:

            for client_id, score in current_similarity_scores.items():

                similarity_history_rows.append(
                    {
                        "Experiment": run_tag,
                        "Attack": attack_type,
                        "Round": round_num + 1,
                        "Client": client_id,
                        "Similarity": score,
                    }
                )

        if current_consistency_scores:

            for client_id, score in current_consistency_scores.items():

                consistency_history_rows.append(
                    {
                        "Experiment": run_tag,
                        "Attack": attack_type,
                        "Round": round_num + 1,
                        "Client": client_id,
                        "Consistency": score,
                    }
                )

        if current_alignment_scores:

            for client_id, score in current_alignment_scores.items():

                alignment_history_rows.append(
                    {
                        "Experiment": run_tag,
                        "Attack": attack_type,
                        "Round": round_num + 1,
                        "Client": client_id,
                        "Alignment": score,
                    }
                )

        if current_norm_scores:

            for client_id, score in current_norm_scores.items():

                norm_history_rows.append(
                    {
                        "Experiment": run_tag,
                        "Attack": attack_type,
                        "Round": round_num + 1,
                        "Client": client_id,
                        "Norm": score,
                    }
                )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                server.get_model().state_dict(),
                f"results/best_model_{run_tag}.pth"
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
        f"results/results_{run_tag}.csv",
        index=False
    )

    with open(
        f"results/summary_{run_tag}.txt",
        "w",
    ) as f:

        f.write(f"Attack Type: {attack_type}\n")
        f.write(f"Malicious Clients: {len(malicious_clients)}\n")
        f.write(f"Malicious Client IDs: {sorted(malicious_clients)}\n")
        f.write(f"Experiment Name: {run_tag}\n")
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
            f"results/trust_history_{run_tag}.csv",
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
            f"results/trust_trajectory_{run_tag}.png",
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
        f"results/accuracy_{run_tag}.png",
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
        f"results/loss_{run_tag}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    if USE_TRUST and trust_history_rows:

        trust_curve_df = pd.DataFrame(trust_history_rows)
        trust_curve_df.to_csv(
            os.path.join(RESULTS_DIR, f"trust_{run_tag}.csv"),
            index=False,
        )

    if behavior_history_rows:

        pd.DataFrame(behavior_history_rows).to_csv(
            os.path.join(RESULTS_DIR, f"behavior_{run_tag}.csv"),
            index=False,
        )

    if similarity_history_rows:

        pd.DataFrame(similarity_history_rows).to_csv(
            os.path.join(RESULTS_DIR, f"similarity_{run_tag}.csv"),
            index=False,
        )

    if consistency_history_rows:

        pd.DataFrame(consistency_history_rows).to_csv(
            os.path.join(RESULTS_DIR, f"consistency_{run_tag}.csv"),
            index=False,
        )

    if alignment_history_rows:

        pd.DataFrame(alignment_history_rows).to_csv(
            os.path.join(RESULTS_DIR, f"alignment_{run_tag}.csv"),
            index=False,
        )

    if norm_history_rows:

        pd.DataFrame(norm_history_rows).to_csv(
            os.path.join(RESULTS_DIR, f"norm_{run_tag}.csv"),
            index=False,
        )

    print("\n========== EXPERIMENT SUMMARY ==========")
    print(f"Attack Type       : {attack_type}")
    print(f"Malicious Clients : {sorted(malicious_clients)}")
    print(f"Best Accuracy     : {best_accuracy:.2f}%")
    print(f"Training Time     : {training_time:.2f} sec")
    print("========================================")

    return {
        "attack_type": attack_type,
        "data_split": DATA_SPLIT,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "attack_scale": ATTACK_SCALE,
        "use_trust": USE_TRUST,
        "use_trust_filter": USE_TRUST_FILTER,
        "use_trimmed_mean": USE_TRIMMED_MEAN,
        "malicious_clients": sorted(malicious_clients),
        "malicious_count": len(malicious_clients),
        "experiment_name": run_tag,
        "final_accuracy": final_accuracy,
        "best_accuracy": best_accuracy,
        "final_loss": final_loss,
        "training_time": training_time,
        "round_diagnostics": round_diagnostics,
        "metrics": metrics,
        "accuracy_history": accuracy_history,
        "loss_history": loss_history,
        "trust_history_rows": trust_history_rows,
        "behavior_history_rows": behavior_history_rows,
        "similarity_history_rows": similarity_history_rows,
        "consistency_history_rows": consistency_history_rows,
        "alignment_history_rows": alignment_history_rows,
        "norm_history_rows": norm_history_rows,
        "filtered_history_rows": filtered_history_rows,
        "final_trust_scores": current_trust_scores,
        "final_behavior_scores": current_behavior_scores,
        "final_similarity_scores": current_similarity_scores,
        "final_consistency_scores": current_consistency_scores,
        "final_alignment_scores": current_alignment_scores,
        "final_norm_scores": current_norm_scores,
        "peer_groups": server.get_peer_groups(),
        "client_histograms": {
            client.client_id: client.label_histogram
            for client in clients
        },
    }


def build_experiment_suite(mode):

    default_weights = (
        SIMILARITY_WEIGHT,
        CONSISTENCY_WEIGHT,
        ALIGNMENT_WEIGHT,
        NORM_WEIGHT,
    )

    comparison_suite = []

    for data_split in ["iid", "noniid", "dirichlet"]:

        for attack_type in ["sign", "scaling", "gaussian", "random", "zero", "mixed"]:

            for use_trust in [False, True]:

                comparison_suite.append(
                    {
                        "name": f"{ 'fedtrust' if use_trust else 'fedavg' }_{data_split}_{attack_type}",
                        "data_split": data_split,
                        "malicious_ratio": 0.2,
                        "attack_type": attack_type,
                        "attack_scale": ATTACK_SCALE,
                        "use_trust": use_trust,
                        "use_trimmed_mean": False,
                        "use_trust_filter": use_trust,
                        "dirichlet_alpha": 0.5,
                        "weights": default_weights,
                    }
                )

    ablation_suite = [
        {
            "name": "full_model",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "mixed",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
        {
            "name": "no_leave_one_out",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "mixed",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": (SIMILARITY_WEIGHT, CONSISTENCY_WEIGHT, 0.0, NORM_WEIGHT),
        },
        {
            "name": "no_mad_norm",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "mixed",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": (SIMILARITY_WEIGHT, CONSISTENCY_WEIGHT, ALIGNMENT_WEIGHT, 0.0),
        },
        {
            "name": "no_trust_filter",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "mixed",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": False,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
    ]

    sensitivity_suite = [
        {
            "name": "weights_4000",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "mixed",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": (0.40, 0.20, 0.20, 0.20),
        },
        {
            "name": "weights_3020",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "mixed",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": (0.30, 0.30, 0.20, 0.20),
        },
        {
            "name": "weights_3525",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "mixed",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": (0.35, 0.25, 0.20, 0.20),
        },
    ]

    heterogeneity_suite = []

    for alpha in [0.1, 0.3, 0.5, 1.0]:

        for use_trust in [False, True]:

            heterogeneity_suite.append(
                {
                    "name": f"{ 'fedtrust' if use_trust else 'fedavg' }_dirichlet_{alpha}",
                    "data_split": "dirichlet",
                    "malicious_ratio": 0.2,
                    "attack_type": "mixed",
                    "attack_scale": ATTACK_SCALE,
                    "use_trust": use_trust,
                    "use_trimmed_mean": False,
                    "use_trust_filter": use_trust,
                    "dirichlet_alpha": alpha,
                    "weights": default_weights,
                }
            )

    stress_suite = [
        {
            "name": "malicious_30pct_scale5",
            "data_split": "dirichlet",
            "malicious_ratio": 0.3,
            "attack_type": "scaling",
            "attack_scale": 5,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
        {
            "name": "malicious_40pct_scale10",
            "data_split": "dirichlet",
            "malicious_ratio": 0.4,
            "attack_type": "scaling",
            "attack_scale": 10,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
        {
            "name": "malicious_40pct_scale20",
            "data_split": "dirichlet",
            "malicious_ratio": 0.4,
            "attack_type": "scaling",
            "attack_scale": 20,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
    ]

    runtime_suite = [
        {
            "name": "runtime_fedavg",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "none",
            "attack_scale": ATTACK_SCALE,
            "use_trust": False,
            "use_trimmed_mean": False,
            "use_trust_filter": False,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
        {
            "name": "runtime_fedtrust",
            "data_split": "dirichlet",
            "malicious_ratio": 0.2,
            "attack_type": "none",
            "attack_scale": ATTACK_SCALE,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
    ]

    diagnostics_suite = [
        {
            "name": "scaling_diagnostic_scale5",
            "data_split": "dirichlet",
            "malicious_ratio": 0.3,
            "attack_type": "scaling",
            "attack_scale": 5,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
        {
            "name": "scaling_diagnostic_scale10",
            "data_split": "dirichlet",
            "malicious_ratio": 0.3,
            "attack_type": "scaling",
            "attack_scale": 10,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
        {
            "name": "scaling_diagnostic_scale20",
            "data_split": "dirichlet",
            "malicious_ratio": 0.3,
            "attack_type": "scaling",
            "attack_scale": 20,
            "use_trust": True,
            "use_trimmed_mean": False,
            "use_trust_filter": True,
            "dirichlet_alpha": 0.5,
            "weights": default_weights,
        },
    ]

    if mode == "comparison":
        return comparison_suite
    if mode == "ablation":
        return ablation_suite
    if mode == "sensitivity":
        return sensitivity_suite
    if mode == "heterogeneity":
        return heterogeneity_suite
    if mode == "stress":
        return stress_suite
    if mode == "runtime":
        return runtime_suite
    if mode == "diagnostics":
        return diagnostics_suite
    if mode == "full":
        return (
            comparison_suite
            + ablation_suite
            + sensitivity_suite
            + heterogeneity_suite
            + stress_suite
            + runtime_suite
            + diagnostics_suite
        )

    raise ValueError(f"Unknown BENCHMARK_MODE: {mode}")


def export_publication_artifacts(summaries):

    summary_lookup = {
        summary["experiment_name"]: summary
        for summary in summaries
    }

    def summary_name(prefix, data_split, attack_type):

        return f"{prefix}_{data_split}_{attack_type}"

    def save_table(base_name, dataframe):

        save_dataframe_artifacts(dataframe, base_name)

    comparison_rows = []

    for data_split, display_name in [
        ("iid", "IID"),
        ("noniid", "Sorted Non-IID"),
        ("dirichlet", f"Dirichlet ({next((summary['dirichlet_alpha'] for summary in summaries if summary['data_split'] == 'dirichlet'), 0.5):.1f})"),
    ]:

        for attack_type in ["none", "sign", "scaling", "gaussian", "random", "zero", "mixed"]:

            fedavg_key = summary_name("fedavg", data_split, attack_type)
            fedtrust_key = summary_name("fedtrust", data_split, attack_type)

            if fedavg_key not in summary_lookup or fedtrust_key not in summary_lookup:

                continue

            fedavg_summary = summary_lookup[fedavg_key]
            fedtrust_summary = summary_lookup[fedtrust_key]

            comparison_rows.append(
                {
                    "Distribution": display_name,
                    "Attack": attack_type.capitalize() if attack_type != "none" else "None",
                    "FedAvg (%)": fedavg_summary["final_accuracy"],
                    "FedTrust (%)": fedtrust_summary["final_accuracy"],
                    "Improvement (%)": fedtrust_summary["final_accuracy"] - fedavg_summary["final_accuracy"],
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        save_table("comparison", comparison_df)
        save_table("table_i_main_comparison", comparison_df)

    runtime_rows = [
        {
            "Method": summary["experiment_name"],
            "Time (sec)": summary["training_time"],
            "Relative Overhead": 0.0,
        }
        for summary in summaries
        if summary["experiment_name"] in {"runtime_fedavg", "runtime_fedtrust"}
    ]

    runtime_df = pd.DataFrame(runtime_rows)
    if not runtime_df.empty:
        baseline_time = runtime_df.loc[runtime_df["Method"] == "runtime_fedavg", "Time (sec)"].iloc[0] if any(runtime_df["Method"] == "runtime_fedavg") else None
        if baseline_time:
            runtime_df["Relative Overhead"] = runtime_df["Time (sec)"] / baseline_time - 1.0
        save_table("runtime", runtime_df)
        save_table("table_ii_runtime", runtime_df)

    ablation_rows = []
    for name in ["full_model", "no_leave_one_out", "no_mad_norm", "no_trust_filter"]:
        summary = summary_lookup.get(name)
        if summary is None:
            continue
        ablation_rows.append(
            {
                "Model": name.replace("_", " ").title(),
                "Mixed Accuracy (%)": summary["final_accuracy"],
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    if not ablation_df.empty:
        save_table("ablation", ablation_df)
        save_table("table_iii_ablation", ablation_df)

    sensitivity_rows = []
    for name in ["weights_4000", "weights_3020", "weights_3525"]:
        summary = summary_lookup.get(name)
        if summary is None:
            continue
        weights = {
            "weights_4000": (0.40, 0.20, 0.20, 0.20),
            "weights_3020": (0.30, 0.30, 0.20, 0.20),
            "weights_3525": (0.35, 0.25, 0.20, 0.20),
        }[name]
        sensitivity_rows.append(
            {
                "Similarity": weights[0],
                "Consistency": weights[1],
                "Alignment": weights[2],
                "Norm": weights[3],
                "Accuracy (%)": summary["final_accuracy"],
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    if not sensitivity_df.empty:
        save_table("weight_sensitivity", sensitivity_df)
        save_table("table_iv_weight_sensitivity", sensitivity_df)

    heterogeneity_rows = []
    for alpha in [0.1, 0.3, 0.5, 1.0]:
        fedavg_summary = summary_lookup.get(f"fedavg_dirichlet_{alpha}")
        fedtrust_summary = summary_lookup.get(f"fedtrust_dirichlet_{alpha}")
        if fedavg_summary is None or fedtrust_summary is None:
            continue
        heterogeneity_rows.append(
            {
                "α": alpha,
                "FedAvg (%)": fedavg_summary["final_accuracy"],
                "FedTrust (%)": fedtrust_summary["final_accuracy"],
                "Improvement (%)": fedtrust_summary["final_accuracy"] - fedavg_summary["final_accuracy"],
            }
        )

    heterogeneity_df = pd.DataFrame(heterogeneity_rows)
    if not heterogeneity_df.empty:
        save_table("heterogeneity", heterogeneity_df)
        save_table("table_v_dirichlet", heterogeneity_df)

    stress_rows = []
    for name in ["malicious_30pct_scale5", "malicious_40pct_scale10", "malicious_40pct_scale20"]:
        summary = summary_lookup.get(name)
        if summary is None:
            continue
        stress_rows.append(
            {
                "Malicious %": int(round(summary["malicious_count"] / NUM_CLIENTS * 100)),
                "Scale": summary["attack_scale"],
                "Accuracy (%)": summary["final_accuracy"],
            }
        )

    stress_df = pd.DataFrame(stress_rows)
    if not stress_df.empty:
        save_table("stress", stress_df)
        save_table("table_vi_stress", stress_df)

    accuracy_rows = []
    loss_rows = []
    trust_rows = []
    behavior_rows = []
    similarity_rows = []
    consistency_rows = []
    alignment_rows = []
    norm_rows = []
    filtered_rows = []

    for summary in summaries:
        accuracy_rows.extend(summary.get("accuracy_history", []))
        loss_rows.extend(summary.get("loss_history", []))
        trust_rows.extend(summary.get("trust_history_rows", []))
        behavior_rows.extend(summary.get("behavior_history_rows", []))
        similarity_rows.extend(summary.get("similarity_history_rows", []))
        consistency_rows.extend(summary.get("consistency_history_rows", []))
        alignment_rows.extend(summary.get("alignment_history_rows", []))
        norm_rows.extend(summary.get("norm_history_rows", []))
        filtered_rows.extend(summary.get("filtered_history_rows", []))

    if accuracy_rows:
        save_dataframe_artifacts(pd.DataFrame(accuracy_rows), "accuracy")
    if loss_rows:
        save_dataframe_artifacts(pd.DataFrame(loss_rows), "loss")
    if trust_rows:
        save_dataframe_artifacts(pd.DataFrame(trust_rows), "trust")
    if behavior_rows:
        save_dataframe_artifacts(pd.DataFrame(behavior_rows), "behavior")
    if similarity_rows:
        save_dataframe_artifacts(pd.DataFrame(similarity_rows), "similarity")
    if consistency_rows:
        save_dataframe_artifacts(pd.DataFrame(consistency_rows), "consistency")
    if alignment_rows:
        save_dataframe_artifacts(pd.DataFrame(alignment_rows), "alignment")
    if norm_rows:
        save_dataframe_artifacts(pd.DataFrame(norm_rows), "norm")
    if filtered_rows:
        save_dataframe_artifacts(pd.DataFrame(filtered_rows), "filtered_clients")

    comparison_for_plot = comparison_df.copy() if not comparison_df.empty else pd.DataFrame()
    if not comparison_for_plot.empty:
        comparison_for_plot["Scenario"] = comparison_for_plot["Distribution"] + "\n" + comparison_for_plot["Attack"]
        save_grouped_bar_chart(
            comparison_for_plot,
            "Scenario",
            ["FedAvg (%)", "FedTrust (%)"],
            ["FedAvg", "FedTrust"],
            "FedAvg vs FedTrust Under Different Attacks",
            "Accuracy (%)",
            "comparison.png",
        )

    if "fedavg_iid_sign" in summary_lookup and "fedtrust_iid_sign" in summary_lookup:
        base_df = pd.DataFrame([
            {
                "Round": round_number + 1,
                "FedAvg Accuracy": fedavg_accuracy,
                "FedTrust Accuracy": fedtrust_accuracy,
            }
            for round_number, (fedavg_accuracy, fedtrust_accuracy) in enumerate(
                zip(
                    summary_lookup["fedavg_iid_sign"]["metrics"]["accuracy"],
                    summary_lookup["fedtrust_iid_sign"]["metrics"]["accuracy"],
                )
            )
        ])
        save_line_plot(
            base_df,
            "Round",
            ["FedAvg Accuracy", "FedTrust Accuracy"],
            ["FedAvg", "FedTrust"],
            "FedAvg vs FedTrust",
            "Accuracy (%)",
            "accuracy.png",
        )

        base_loss_df = pd.DataFrame([
            {
                "Round": round_number + 1,
                "FedAvg Loss": fedavg_loss,
                "FedTrust Loss": fedtrust_loss,
            }
            for round_number, (fedavg_loss, fedtrust_loss) in enumerate(
                zip(
                    summary_lookup["fedavg_iid_sign"]["metrics"]["loss"],
                    summary_lookup["fedtrust_iid_sign"]["metrics"]["loss"],
                )
            )
        ])
        save_line_plot(
            base_loss_df,
            "Round",
            ["FedAvg Loss", "FedTrust Loss"],
            ["FedAvg", "FedTrust"],
            "FedAvg vs FedTrust",
            "Loss",
            "loss.png",
        )

    if "fedtrust_dirichlet_mixed" in summary_lookup:
        mixed_summary = summary_lookup["fedtrust_dirichlet_mixed"]
        if mixed_summary.get("trust_history_rows"):
            trust_curve_df = pd.DataFrame(mixed_summary["trust_history_rows"])
            trust_curve_df = trust_curve_df.pivot(
                index="Round",
                columns="Client",
                values="Trust",
            ).reset_index()
            trust_columns = [column for column in trust_curve_df.columns if column != "Round"]
            save_line_plot(
                trust_curve_df,
                "Round",
                trust_columns,
                [f"Client {column}" for column in trust_columns],
                "FedTrust Mixed Attack Trajectory",
                "Trust",
                "trust.png",
            )

        filtered_curve = pd.DataFrame([
            {
                "Round": index + 1,
                "Filtered Clients": row.get("Filtered Clients", 0),
            }
            for index, row in enumerate(mixed_summary["filtered_history_rows"])
        ])
        if not filtered_curve.empty:
            save_line_plot(
                filtered_curve,
                "Round",
                ["Filtered Clients"],
                ["Filtered Clients"],
                "Number of Filtered Clients",
                "Filtered Clients",
                "filtered_clients.png",
            )

        if mixed_summary.get("final_trust_scores"):
            trust_hist_labels = [f"Client {client_id}" for client_id in sorted(mixed_summary["final_trust_scores"].keys())]
            trust_hist_values = [mixed_summary["final_trust_scores"][client_id] for client_id in sorted(mixed_summary["final_trust_scores"].keys())]
            save_bar_chart(
                trust_hist_labels,
                trust_hist_values,
                "Histogram of Trust Scores - Last Round",
                "Trust",
                "trust_histogram.png",
                rotation=30,
            )

        if mixed_summary.get("final_behavior_scores"):
            behavior_labels = [f"Client {client_id}" for client_id in sorted(mixed_summary["final_behavior_scores"].keys())]
            behavior_values = [mixed_summary["final_behavior_scores"][client_id] for client_id in sorted(mixed_summary["final_behavior_scores"].keys())]
            save_bar_chart(
                behavior_labels,
                behavior_values,
                "Behavior Score - Round 20",
                "Behavior",
                "behavior.png",
                rotation=30,
            )

        if mixed_summary.get("client_histograms"):
            save_stacked_dirichlet_chart(
                mixed_summary["client_histograms"],
                "Dirichlet Distribution Across Clients",
                "dirichlet_distribution.png",
            )

        if mixed_summary.get("peer_groups"):
            peer_membership = {}
            for group_id, members in mixed_summary["peer_groups"].items():
                for member in members:
                    peer_membership[member] = group_id

            peer_labels = [f"C{client_id}" for client_id in sorted(peer_membership.keys())]
            peer_values = [peer_membership[client_id] for client_id in sorted(peer_membership.keys())]
            save_bar_chart(
                peer_labels,
                peer_values,
                "Peer Groups",
                "Peer Group ID",
                "peer_groups.png",
                rotation=30,
            )

        save_flow_diagram(
            [
                "Client",
                "Local Training",
                "Behavior Score",
                "Trust",
                "Trust Filter",
                "Aggregation",
                "Global Model",
            ],
            "Federated Trust Framework",
            "framework.png",
        )

    if "fedtrust_iid_random" in summary_lookup:
        random_summary = summary_lookup["fedtrust_iid_random"]
        random_df = pd.DataFrame([
            {
                "Round": row["Round"],
                "Accuracy": row["Accuracy"],
            }
            for row in random_summary["accuracy_history"]
        ])
        if not random_df.empty:
            save_line_plot(
                random_df,
                "Round",
                ["Accuracy"],
                ["FedTrust"],
                "Random Attack Accuracy",
                "Accuracy (%)",
                "random_attack_accuracy.png",
            )

    if "fedtrust_iid_gaussian" in summary_lookup:
        gaussian_summary = summary_lookup["fedtrust_iid_gaussian"]
        gaussian_df = pd.DataFrame([
            {
                "Round": row["Round"],
                "Accuracy": row["Accuracy"],
            }
            for row in gaussian_summary["accuracy_history"]
        ])
        if not gaussian_df.empty:
            save_line_plot(
                gaussian_df,
                "Round",
                ["Accuracy"],
                ["FedTrust"],
                "Gaussian Attack Accuracy",
                "Accuracy (%)",
                "gaussian_accuracy.png",
            )

    if "fedtrust_iid_scaling" in summary_lookup:
        scaling_summary = summary_lookup["fedtrust_iid_scaling"]
        scaling_df = pd.DataFrame([
            {
                "Round": row["Round"],
                "Loss": row["Loss"],
            }
            for row in scaling_summary["loss_history"]
        ])
        if not scaling_df.empty:
            save_line_plot(
                scaling_df,
                "Round",
                ["Loss"],
                ["FedTrust"],
                "Scaling Attack Loss Explosion",
                "Loss",
                "scaling_loss.png",
            )



def run_benchmark_suite(mode):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    benchmark_rows = []

    for config in build_experiment_suite(mode):

        set_experiment_params(
            data_split=config["data_split"],
            dirichlet_alpha=config["dirichlet_alpha"],
            attack_scale=config["attack_scale"],
            use_trust=config["use_trust"],
            use_trimmed_mean=config["use_trimmed_mean"],
            use_trust_filter=config["use_trust_filter"],
            similarity_weight=config["weights"][0],
            consistency_weight=config["weights"][1],
            alignment_weight=config["weights"][2],
            norm_weight=config["weights"][3],
        )

        malicious_clients = select_malicious_clients(
            NUM_CLIENTS,
            config["malicious_ratio"],
            seed_offset=len(benchmark_rows),
        )

        summary = run_experiment(
            config["attack_type"],
            malicious_clients,
            experiment_name=config["name"],
            collect_diagnostics=True,
        )

        benchmark_rows.append(summary)

    comparison_df = pd.DataFrame([
        {
            "Experiment": row["experiment_name"],
            "Data Split": row["data_split"],
            "Attack": row["attack_type"],
            "Dirichlet Alpha": row["dirichlet_alpha"],
            "Attack Scale": row["attack_scale"],
            "Malicious Clients": row["malicious_count"],
            "Trust Enabled": row["use_trust"],
            "Trust Filter": row["use_trust_filter"],
            "Best Accuracy": row["best_accuracy"],
            "Final Accuracy": row["final_accuracy"],
            "Final Loss": row["final_loss"],
            "Training Time": row["training_time"],
            "Filtered Diagnostics": "yes" if row["round_diagnostics"] else "no",
        }
        for row in benchmark_rows
    ])

    comparison_df.to_csv(
        os.path.join(RESULTS_DIR, "benchmark_comparison.csv"),
        index=False,
    )

    if any(row["round_diagnostics"] for row in benchmark_rows):
        diagnostics_rows = []

        for row in benchmark_rows:
            diagnostics_rows.extend(row["round_diagnostics"])

        pd.DataFrame(diagnostics_rows).to_csv(
            os.path.join(RESULTS_DIR, "benchmark_diagnostics.csv"),
            index=False,
        )

    export_publication_artifacts(benchmark_rows)

    return benchmark_rows


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if RUN_ALL_ATTACKS:

        summaries = run_benchmark_suite(BENCHMARK_MODE)

    else:

        rng = random.Random(SEED)
        num_malicious = min(MALICIOUS_CLIENTS, NUM_CLIENTS)

        malicious_clients = set(
            rng.sample(
                range(NUM_CLIENTS),
                num_malicious
            )
        )

        summaries = [
            run_experiment(
                "mixed",
                malicious_clients,
                experiment_name=EXPERIMENT_TAG,
                collect_diagnostics=SAVE_DIAGNOSTICS,
            )
        ]

    print("\n========== ALL ATTACKS COMPLETE ==========")

    for summary in summaries:

        print(
            f"{summary['attack_type']:>8} : best={summary['best_accuracy']:.2f}% "
            f"time={summary['training_time']:.2f} sec"
        )

    summary_rows = pd.DataFrame([
        {
            "Experiment": summary["experiment_name"],
            "Data Split": summary["data_split"],
            "Attack": summary["attack_type"],
            "Trust Enabled": summary["use_trust"],
            "Trust Filter": summary["use_trust_filter"],
            "Dirichlet Alpha": summary["dirichlet_alpha"] if summary["data_split"] == "dirichlet" else "N/A",
            "Attack Scale": summary["attack_scale"],
            "Malicious Clients": summary["malicious_count"],
            "Malicious Client IDs": str(summary["malicious_clients"]),
            "Best Accuracy": summary["best_accuracy"],
            "Final Accuracy": summary["final_accuracy"],
            "Final Loss": summary["final_loss"],
            "Training Time": summary["training_time"],
        }
        for summary in summaries
    ])

    summary_rows.to_csv(
        os.path.join(RESULTS_DIR, "experiment_summary.csv"),
        index=False
    )

    if summaries and any(summary["round_diagnostics"] for summary in summaries):

        diagnostics_rows = []

        for summary in summaries:

            for item in summary["round_diagnostics"]:

                diagnostics_rows.append(
                    {
                        "Experiment": summary["experiment_name"],
                        **item,
                    }
                )

        pd.DataFrame(diagnostics_rows).to_csv(
            os.path.join(RESULTS_DIR, "run_diagnostics.csv"),
            index=False,
        )

    print("========================================")


if __name__ == "__main__":

    main()