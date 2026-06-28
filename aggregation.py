import copy
import torch


def fedavg(client_updates):

    total_samples = sum(
        client["num_samples"]
        for client in client_updates
    )

    avg_weights = copy.deepcopy(
        client_updates[0]["weights"]
    )

    # Initialize weighted sum
    for key in avg_weights.keys():

        avg_weights[key] *= (
            client_updates[0]["num_samples"]
            / total_samples
        )

    # Add remaining clients
    for client in client_updates[1:]:

        weight = (
            client["num_samples"]
            / total_samples
        )

        for key in avg_weights.keys():

            avg_weights[key] += (
                client["weights"][key] * weight
            )

    return avg_weights